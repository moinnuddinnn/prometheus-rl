"""Online DQN training and isolated evaluation for CartPole-v1."""

import csv
import json
import platform
import subprocess
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from importlib.metadata import PackageNotFoundError, version
from math import isfinite
from pathlib import Path
from random import Random
from statistics import mean, stdev
from time import perf_counter
from typing import Protocol

from prometheus_rl.agents.dqn import DQNAgent, DQNConfig
from prometheus_rl.memory import Transition


class Environment(Protocol):
    def reset(self, *, seed: int | None) -> tuple[Sequence[float], dict]: ...
    def step(self, action: int) -> tuple[Sequence[float], float, bool, bool, dict]: ...
    def close(self) -> None: ...


Factory = Callable[[], Environment]


@dataclass(frozen=True, slots=True)
class TrainConfig:
    steps: int = 100_000
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    decay_steps: int = 30_000
    train_every: int = 4
    seed: int = 0

    def __post_init__(self) -> None:
        if any(type(x) is not int or x < 1 for x in (
            self.steps, self.decay_steps, self.train_every
        )):
            raise ValueError("step counts must be positive integers")
        if type(self.seed) is not int or not 0 <= self.seed < 2**32:
            raise ValueError("seed must be a nonnegative 32-bit integer")
        if not (isfinite(self.epsilon_start) and isfinite(self.epsilon_end)
                and 0 <= self.epsilon_end <= self.epsilon_start <= 1):
            raise ValueError("require 0 <= epsilon_end <= epsilon_start <= 1")

    def epsilon(self, step: int) -> float:
        fraction = min(1.0, max(0.0, step / self.decay_steps))
        return self.epsilon_start + fraction * (self.epsilon_end - self.epsilon_start)


@dataclass(frozen=True, slots=True)
class Episode:
    episode: int
    step: int
    reward: float
    length: int
    epsilon: float
    mean_loss: float | None
    terminated: bool
    truncated: bool


@dataclass(frozen=True, slots=True)
class Evaluation:
    returns: tuple[float, ...]

    @property
    def average(self) -> float:
        return mean(self.returns)

    @property
    def std(self) -> float:
        return stdev(self.returns) if len(self.returns) > 1 else 0.0


def make_cartpole(*, render_mode: str | None = None) -> Environment:
    import gymnasium as gym

    return gym.make("CartPole-v1", render_mode=render_mode)


def train(
    agent: DQNAgent, config: TrainConfig, *, factory: Factory = make_cartpole,
    on_episode: Callable[[Episode], None] | None = None,
) -> tuple[Episode, ...]:
    """Collect exactly config.steps transitions; discard incomplete episode metrics."""
    env = factory()
    records = []
    try:
        obs, _ = env.reset(seed=config.seed)
        # Snapshot observations in case the environment reuses an array.
        obs = tuple(float(x) for x in obs)
        reward_sum = 0.0
        length = 0
        losses = []
        for step in range(1, config.steps + 1):
            epsilon = config.epsilon(step - 1)
            action = agent.act(obs, epsilon=epsilon)
            nxt, reward, terminated, truncated, _ = env.step(action)
            transition = Transition(obs, action, reward, nxt, terminated, truncated)
            agent.observe(transition)
            if step % config.train_every == 0:
                update = agent.update()
                if update is not None:
                    losses.append(update.loss)
            reward_sum += reward
            length += 1
            obs = transition.next_observation
            if transition.episode_end:
                record = Episode(len(records) + 1, step, reward_sum, length,
                                 epsilon, mean(losses) if losses else None,
                                 terminated, truncated)
                records.append(record)
                if on_episode is not None:
                    on_episode(record)
                if step < config.steps:
                    nxt, _ = env.reset(seed=None)
                    obs = tuple(float(x) for x in nxt)
                reward_sum, length, losses = 0.0, 0, []
    finally:
        env.close()
    return tuple(records)


def evaluate(
    agent: DQNAgent, *, episodes: int = 100, seed: int = 1_000_000,
    random_policy: bool = False, factory: Factory = make_cartpole,
) -> Evaluation:
    """Frozen greedy or uniform-random policy on separate environment seeds."""
    if type(episodes) is not int or episodes < 1:
        raise ValueError("episodes must be a positive integer")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = Random(seed)
    env = factory()
    returns = []
    try:
        for i in range(episodes):
            obs, _ = env.reset(seed=seed + i)
            total = 0.0
            while True:
                action = (rng.randrange(agent.config.action_dim) if random_policy
                          else agent.act(obs, epsilon=0.0))
                obs, reward, terminated, truncated, _ = env.step(action)
                total += reward
                if terminated or truncated:
                    break
            returns.append(total)
    finally:
        env.close()
    return Evaluation(tuple(returns))


def _commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _versions() -> dict[str, str]:
    result = {"python": platform.python_version()}
    for name in ("torch", "gymnasium", "numpy", "matplotlib", "prometheus-rl"):
        try:
            result[name] = version(name)
        except PackageNotFoundError:
            result[name] = "not installed"
    return result


def write_results(path: Path, records: Sequence[Episode], metadata: dict) -> None:
    with (path / "episodes.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(Episode.__dataclass_fields__))
        writer.writeheader()
        writer.writerows(asdict(record) for record in records)
    (path / "summary.json").write_text(json.dumps(metadata, indent=2) + "\n")


def plot_results(path: Path, records: Sequence[Episode]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps = [r.step for r in records]
    returns = [r.reward for r in records]
    rolling = [mean(returns[max(0, i - 19):i + 1]) for i in range(len(returns))]
    fig, ax = plt.subplots()
    ax.plot(steps, returns, alpha=0.35, label="Episode return")
    ax.plot(steps, rolling, label="Rolling mean (20 episodes)")
    ax.set(xlabel="Environment steps", ylabel="Undiscounted return")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path / "returns.png", dpi=150)
    plt.close(fig)
    fig, ax = plt.subplots()
    valid = [r for r in records if r.mean_loss is not None]
    ax.plot([r.step for r in valid], [r.mean_loss for r in valid])
    ax.set(xlabel="Environment steps", ylabel="Mean Huber loss per episode")
    fig.tight_layout()
    fig.savefig(path / "loss.png", dpi=150)
    plt.close(fig)


def run_experiment(
    path: Path, config: TrainConfig, *, eval_episodes: int = 100,
    eval_seed: int = 1_000_000, device: str = "cpu", plots: bool = True,
    dqn_config: DQNConfig | None = None,
) -> dict:
    """Train once, then evaluate final weights against a random baseline."""
    if type(eval_episodes) is not int or eval_episodes < 1:
        raise ValueError("eval_episodes must be positive")
    if type(eval_seed) is not int or eval_seed < 0:
        raise ValueError("eval_seed must be nonnegative")
    # Conservatively reserve a disjoint numerical seed range for evaluation.
    if not (eval_seed >= config.seed + config.steps + 1
            or eval_seed + eval_episodes <= config.seed):
        raise ValueError("training and evaluation seed ranges must not overlap")
    import torch

    # Fail before creating a run directory if the environment is unavailable.
    probe = make_cartpole()
    probe.close()
    path.mkdir(parents=True, exist_ok=False)
    start = perf_counter()
    agent = DQNAgent(dqn_config or DQNConfig(), seed=config.seed, device=device)

    def progress(record: Episode) -> None:
        if record.episode % 20 == 0:
            print(f"episode={record.episode} step={record.step} "
                  f"return={record.reward:.0f} epsilon={record.epsilon:.3f}",
                  flush=True)

    records = train(agent, config, on_episode=progress)
    agent.save(path / "model.pt")
    greedy = evaluate(agent, episodes=eval_episodes, seed=eval_seed)
    random = evaluate(agent, episodes=eval_episodes, seed=eval_seed,
                      random_policy=True)
    summary = {
        "environment": "CartPole-v1", "device": device, "commit": _commit(),
        "versions": _versions(), "torch_threads": torch.get_num_threads(),
        "training": asdict(config),
        "dqn": asdict(agent.config), "updates": agent.updates,
        "completed_episodes": len(records), "evaluation_seed": eval_seed,
        "evaluation_episodes": eval_episodes, "seconds": perf_counter() - start,
        "greedy": {"returns": greedy.returns, "mean": greedy.average,
                   "std": greedy.std},
        "random": {"returns": random.returns, "mean": random.average,
                   "std": random.std},
    }
    write_results(path, records, summary)
    if plots:
        plot_results(path, records)
    return summary
