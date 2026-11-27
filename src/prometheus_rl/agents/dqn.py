"""Vanilla DQN with detached target estimates and uniform replay."""

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from math import isfinite
from pathlib import Path
from random import Random

import torch
from torch import Tensor

from prometheus_rl.memory import ReplayBuffer, Transition
from prometheus_rl.memory.batch import TransitionBatch, collate
from prometheus_rl.networks import QNetwork, fixed_target_loss


@dataclass(frozen=True, slots=True)
class DQNConfig:
    observation_dim: int = 4
    action_dim: int = 2
    hidden_dims: tuple[int, ...] = (128, 128)
    gamma: float = 0.99
    learning_rate: float = 1e-3
    capacity: int = 50_000
    batch_size: int = 64
    warmup: int = 1_000
    target_interval: int = 250
    max_grad_norm: float = 10.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "hidden_dims", tuple(self.hidden_dims))
        dimensions = (self.observation_dim, self.action_dim, *self.hidden_dims)
        counts = (self.capacity, self.batch_size, self.warmup, self.target_interval)
        if any(type(x) is not int or x <= 0 for x in (*dimensions, *counts)):
            raise ValueError("dimensions and counts must be positive integers")
        if not self.batch_size <= self.warmup <= self.capacity:
            raise ValueError("require batch_size <= warmup <= capacity")
        if not isfinite(self.gamma) or not 0 <= self.gamma < 1:
            raise ValueError("require 0 <= gamma < 1")
        if any(not isfinite(x) or x <= 0
               for x in (self.learning_rate, self.max_grad_norm)):
            raise ValueError("learning rate and gradient bound must be positive")


@dataclass(frozen=True, slots=True)
class UpdateResult:
    loss: float
    grad_norm: float
    updates: int
    target_synced: bool


@torch.no_grad()
def bellman_targets(target: QNetwork, batch: TransitionBatch, gamma: float) -> Tensor:
    """Bootstrap only nonterminal rows, including external truncations."""
    if not isfinite(gamma) or not 0 <= gamma < 1:
        raise ValueError("require 0 <= gamma < 1")
    result = batch.rewards.clone()
    active = ~batch.terminated
    if gamma and active.any():
        result[active] += gamma * target(batch.next_observations[active]).amax(1)
    return result


class DQNAgent:
    """Own the networks, optimizer, behavior RNG and replay; no environment API."""

    def __init__(self, config: DQNConfig, *, seed: int = 0,
                 device: str | torch.device = "cpu") -> None:
        self.config = config
        self.device = torch.device(device)
        self._rng = Random(seed)
        # Initialize on CPU without altering the caller's torch RNG stream.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            dims = (config.observation_dim, config.action_dim, config.hidden_dims)
            self.online = QNetwork(*dims).to(self.device)
            self.target = QNetwork(*dims).to(self.device)
        self.target.requires_grad_(False)
        self.target.eval()
        self.sync_target()
        self.optimizer = torch.optim.Adam(self.online.parameters(),
                                          lr=config.learning_rate)
        self.replay = ReplayBuffer(config.capacity, config.observation_dim,
                                   config.action_dim, seed=seed + 1)
        self.updates = 0

    @torch.no_grad()
    def act(self, observation: Sequence[float], *, epsilon: float = 0.0) -> int:
        if not isfinite(epsilon) or not 0 <= epsilon <= 1:
            raise ValueError("epsilon must be in [0, 1]")
        x = torch.tensor(tuple(observation), dtype=torch.float32, device=self.device)
        if x.shape != (self.config.observation_dim,) or not torch.isfinite(x).all():
            raise ValueError("invalid observation")
        if epsilon and self._rng.random() < epsilon:
            return self._rng.randrange(self.config.action_dim)
        return int(self.online(x).argmax().item())

    def observe(self, transition: Transition) -> None:
        self.replay.append(transition)

    def sync_target(self) -> None:
        self.target.load_state_dict(self.online.state_dict())

    def update(self) -> UpdateResult | None:
        """One Adam step after warm-up; synchronize by optimizer-step count."""
        cfg = self.config
        if len(self.replay) < cfg.warmup:
            return None
        batch = collate(self.replay.sample(cfg.batch_size), device=self.device)
        targets = bellman_targets(self.target, batch, cfg.gamma)
        loss = fixed_target_loss(
            self.online(batch.observations), batch.actions, targets
        )
        if not torch.isfinite(loss):
            raise FloatingPointError("nonfinite DQN loss")
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(
            self.online.parameters(), cfg.max_grad_norm, error_if_nonfinite=True
        )
        self.optimizer.step()
        self.updates += 1
        synced = self.updates % cfg.target_interval == 0
        if synced:
            self.sync_target()
        return UpdateResult(float(loss.detach()), float(norm), self.updates, synced)

    def save(self, path: str | Path) -> None:
        """Save model/optimizer state, not replay or environment/RNG state."""
        torch.save({"format": 1, "config": asdict(self.config),
                    "online": self.online.state_dict(),
                    "target": self.target.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    "updates": self.updates}, path)

    @classmethod
    def load(cls, path: str | Path, *, seed: int = 0,
             device: str | torch.device = "cpu") -> "DQNAgent":
        """Restore for inference or warm-start training with an empty buffer."""
        data = torch.load(path, map_location=device, weights_only=True)
        if data["format"] != 1:
            raise ValueError("unsupported checkpoint format")
        agent = cls(DQNConfig(**data["config"]), seed=seed, device=device)
        agent.online.load_state_dict(data["online"])
        agent.target.load_state_dict(data["target"])
        agent.optimizer.load_state_dict(data["optimizer"])
        agent.updates = data["updates"]
        return agent
