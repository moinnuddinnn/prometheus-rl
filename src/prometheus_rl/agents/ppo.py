"""On-policy actor-critic, generalized advantages and clipped PPO updates."""

import math
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True, slots=True)
class PPOConfig:
    observation_dim: int
    action_dim: int
    continuous: bool = False
    hidden: int = 64
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip: float = 0.2
    learning_rate: float = 0.0003
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    epochs: int = 4
    batch_size: int = 64

    def __post_init__(self):
        counts = (self.observation_dim, self.action_dim, self.hidden,
                  self.epochs, self.batch_size)
        if any(type(n) is not int or n < 1 for n in counts):
            raise ValueError("dimensions and counts must be positive integers")
        rates = (self.gamma, self.gae_lambda, self.clip, self.learning_rate,
                 self.value_coef, self.entropy_coef, self.max_grad_norm)
        if not all(math.isfinite(x) for x in rates):
            raise ValueError("nonfinite configuration")
        if not (0 <= self.gamma < 1 and 0 <= self.gae_lambda <= 1
                and 0 < self.clip < 1 and self.learning_rate > 0
                and self.value_coef >= 0 and self.entropy_coef >= 0
                and self.max_grad_norm > 0):
            raise ValueError("invalid PPO hyperparameters")


class ActorCritic(nn.Module):
    """Categorical or tanh-squashed Gaussian policy, with scalar critic."""

    def __init__(self, config: PPOConfig):
        super().__init__()
        self.config = config
        self.body = nn.Sequential(
            nn.Linear(config.observation_dim, config.hidden), nn.Tanh(),
            nn.Linear(config.hidden, config.hidden), nn.Tanh(),
        )
        self.actor = nn.Linear(config.hidden, config.action_dim)
        self.critic = nn.Linear(config.hidden, 1)
        if config.continuous:
            self.log_std = nn.Parameter(torch.zeros(config.action_dim))

    def forward(self, obs: Tensor) -> tuple[Tensor, Tensor]:
        features = self.body(obs)
        return self.actor(features), self.critic(features).squeeze(-1)

    def statistics(self, obs: Tensor, action: Tensor):
        """Log probability of stored action (latent action for Gaussian)."""
        logits, values = self(obs)
        if self.config.continuous:
            dist = torch.distributions.Normal(logits, self.log_std.clamp(-5, 2).exp())
            # Stable log(1-tanh(z)^2); use pre-squash samples, never atanh.
            jac = 2 * (math.log(2) - action - F.softplus(-2 * action))
            logp = (dist.log_prob(action) - jac).sum(-1)
            # Base Gaussian entropy is an exploration proxy, not squashed entropy.
            entropy = dist.entropy().sum(-1)
        else:
            dist = torch.distributions.Categorical(logits=logits)
            logp, entropy = dist.log_prob(action), dist.entropy()
        return logp, entropy, values

    @torch.no_grad()
    def sample(self, obs: Tensor, rng: torch.Generator, *, greedy=False):
        logits, value = self(obs)
        if self.config.continuous:
            action = logits if greedy else logits + self.log_std.clamp(-5, 2).exp() * (
                torch.randn(logits.shape, generator=rng, device=logits.device)
            )
        else:
            action = logits.argmax(-1) if greedy else torch.multinomial(
                logits.softmax(-1), 1, generator=rng
            ).squeeze(-1)
        logp, _, _ = self.statistics(obs, action)
        return action, logp, value


def advantages(rewards: Tensor, values: Tensor, next_values: Tensor,
               terminated: Tensor, truncated: Tensor, gamma: float,
               lam: float) -> tuple[Tensor, Tensor]:
    """Bootstrap truncations but never propagate advantages across resets."""
    if rewards.ndim != 1 or not rewards.numel() or any(
        x.shape != rewards.shape for x in (values, next_values, terminated, truncated)
    ):
        raise ValueError("rollout tensors must be nonempty matching vectors")
    delta = rewards + gamma * (~terminated.bool()) * next_values - values
    carry = torch.zeros_like(delta[0])
    result = torch.empty_like(delta)
    for t in reversed(range(len(delta))):
        carry = delta[t] + gamma * lam * (~(terminated[t].bool() |
                                           truncated[t].bool())) * carry
        result[t] = carry
    return result, result + values


def discounted_returns(rewards: Tensor, gamma: float) -> Tensor:
    """Monte Carlo reward-to-go for a complete finite trajectory."""
    if rewards.ndim != 1 or not rewards.numel() or not 0 <= gamma <= 1:
        raise ValueError("invalid reward vector or discount")
    result = torch.empty_like(rewards)
    carry = torch.zeros_like(rewards[0])
    for t in reversed(range(len(rewards))):
        carry = rewards[t] + gamma * carry
        result[t] = carry
    return result


def reinforce_loss(log_prob: Tensor, discounted_returns: Tensor) -> Tensor:
    """Monte Carlo score-function objective, with detached return weights."""
    return -(log_prob * discounted_returns.detach()).mean()


def clipped_loss(logp: Tensor, old_logp: Tensor, advantage: Tensor,
                 clip: float) -> Tensor:
    ratio = (logp - old_logp.detach()).exp()
    a = advantage.detach()
    return -torch.minimum(ratio * a, ratio.clamp(1 - clip, 1 + clip) * a).mean()


@dataclass(slots=True)
class Rollout:
    observations: Tensor
    actions: Tensor
    old_logp: Tensor
    advantages: Tensor
    returns: Tensor


class PPOAgent:
    def __init__(self, config: PPOConfig, *, seed=0, device="cpu"):
        self.config = config
        self.device = torch.device(device)
        # Isolate initialization from callers' CPU random state.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.network = ActorCritic(config).to(self.device)
        self.rng = torch.Generator(device=self.device).manual_seed(seed)
        self.optimizer = torch.optim.Adam(self.network.parameters(),
                                          lr=config.learning_rate)
        self.updates = 0

    def update(self, rollout: Rollout) -> dict[str, float]:
        cfg = self.config
        n = len(rollout.observations)
        if n < 1:
            raise ValueError("empty rollout")
        adv = rollout.advantages.detach()
        # A singleton or constant-advantage batch must retain its learning signal.
        if n > 1 and adv.std(unbiased=False).item() > 1e-8:
            adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)
        totals = torch.zeros(4, device=self.device)
        count = 0
        for _ in range(cfg.epochs):
            order = torch.randperm(n, generator=self.rng, device=self.device)
            for i in order.split(cfg.batch_size):
                logp, entropy, value = self.network.statistics(
                    rollout.observations[i], rollout.actions[i]
                )
                policy = clipped_loss(logp, rollout.old_logp[i], adv[i], cfg.clip)
                critic = F.mse_loss(value, rollout.returns[i].detach())
                ent = entropy.mean()
                loss = policy + cfg.value_coef * critic - cfg.entropy_coef * ent
                if not torch.isfinite(loss):
                    raise FloatingPointError("nonfinite PPO loss")
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), cfg.max_grad_norm,
                                         error_if_nonfinite=True)
                self.optimizer.step()
                self.updates += 1
                with torch.no_grad():
                    kl = (rollout.old_logp[i] - logp).mean()
                    totals += torch.stack((policy, critic, ent, kl)).detach() * len(i)
                count += len(i)
        return dict(zip(("policy_loss", "value_loss", "entropy", "sample_kl"),
                        (totals / count).tolist(), strict=True))

    def save(self, path: Path, *, game: str):
        torch.save({"config": asdict(self.config), "network": self.network.state_dict(),
                    "optimizer": self.optimizer.state_dict(), "updates": self.updates,
                    "game": game}, path)

    @classmethod
    def load(cls, path: Path, *, device="cpu"):
        state = torch.load(path, map_location=device, weights_only=True)
        agent = cls(PPOConfig(**state["config"]), device=device)
        agent.network.load_state_dict(state["network"])
        agent.optimizer.load_state_dict(state["optimizer"])
        agent.updates = state["updates"]
        return agent, state["game"]
