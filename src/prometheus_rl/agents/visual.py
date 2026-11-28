"""Visual DQN with uniform byte replay and optional Double/Dueling variants."""

from dataclasses import asdict, dataclass
from math import isfinite
from pathlib import Path
from random import Random

import numpy as np
import torch
from torch.nn import functional as F

from prometheus_rl.agents.dqn import DQNConfig, UpdateResult, bellman_targets
from prometheus_rl.memory.pixels import PixelExperience, PixelReplay, validate_pixels
from prometheus_rl.networks.visual import VisualQNetwork


@dataclass(frozen=True, slots=True)
class VisualConfig:
    action_dim: int
    capacity: int = 5000
    batch_size: int = 32
    warmup: int = 1000
    gamma: float = 0.99
    learning_rate: float = 1e-4
    target_interval: int = 1000
    double: bool = True
    dueling: bool = True

    def __post_init__(self) -> None:
        # Reuse the scalar validation contract; the visual input is fixed.
        DQNConfig(action_dim=self.action_dim, capacity=self.capacity,
                  batch_size=self.batch_size, warmup=self.warmup,
                  gamma=self.gamma, learning_rate=self.learning_rate,
                  target_interval=self.target_interval, double=self.double,
                  dueling=self.dueling)


class VisualAgent:
    def __init__(self, config: VisualConfig, *, seed: int = 0,
                 device: str = "cpu") -> None:
        self.config = config
        self.device = torch.device(device)
        self._rng = Random(seed)
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            self.online = VisualQNetwork(config.action_dim, dueling=config.dueling)
            self.target = VisualQNetwork(config.action_dim, dueling=config.dueling)
        self.online.to(self.device)
        self.target.to(self.device).requires_grad_(False).eval()
        self.target.load_state_dict(self.online.state_dict())
        self.optimizer = torch.optim.Adam(self.online.parameters(),
                                          lr=config.learning_rate)
        self.replay = PixelReplay(config.capacity, config.action_dim, seed=seed + 1)
        self.updates = 0

    @torch.no_grad()
    def act(self, observation: np.ndarray, *, epsilon: float = 0.0) -> int:
        obs = validate_pixels(observation)
        if not isfinite(epsilon) or not 0 <= epsilon <= 1:
            raise ValueError("epsilon must be in [0, 1]")
        if epsilon and self._rng.random() < epsilon:
            return self._rng.randrange(self.config.action_dim)
        x = torch.tensor(obs, device=self.device).unsqueeze(0)
        return int(self.online(x).argmax(dim=1).item())

    def observe(self, transition: PixelExperience) -> None:
        self.replay.append(transition)

    def update(self) -> UpdateResult | None:
        c = self.config
        if len(self.replay) < c.warmup:
            return None
        batch = self.replay.sample(c.batch_size, device=self.device)
        targets = bellman_targets(self.target, batch, c.gamma,
                                  online=self.online if c.double else None)
        chosen = self.online(batch.observations).gather(
            1, batch.actions[:, None]
        ).squeeze(1)
        loss = F.smooth_l1_loss(chosen, targets)
        if not torch.isfinite(loss):
            raise FloatingPointError("nonfinite visual DQN loss")
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(self.online.parameters(), 10.0,
                                              error_if_nonfinite=True)
        self.optimizer.step()
        self.updates += 1
        synced = self.updates % c.target_interval == 0
        if synced:
            self.target.load_state_dict(self.online.state_dict())
        return UpdateResult(float(loss.detach()), float(norm), self.updates, synced)

    def save(self, path: str | Path, *, game: str) -> None:
        torch.save({"format": "visual-1", "game": game,
                    "config": asdict(self.config), "updates": self.updates,
                    "online": self.online.state_dict(),
                    "target": self.target.state_dict(),
                    "optimizer": self.optimizer.state_dict()}, path)

    @classmethod
    def load(cls, path: str | Path, *, device: str = "cpu"):
        data = torch.load(path, map_location=device, weights_only=True)
        if data["format"] != "visual-1":
            raise ValueError("not a visual checkpoint")
        agent = cls(VisualConfig(**data["config"]), device=device)
        agent.online.load_state_dict(data["online"])
        agent.target.load_state_dict(data["target"])
        agent.optimizer.load_state_dict(data["optimizer"])
        agent.updates = data["updates"]
        return agent, data["game"]
