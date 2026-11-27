"""Seeded training and held-out evaluation for finite environments."""

from dataclasses import dataclass
from math import fsum, isfinite
from random import Random

from prometheus_rl.agents import QLearning
from prometheus_rl.core import FiniteMDP


@dataclass(frozen=True, slots=True)
class Episode:
    discounted_return: float
    steps: int
    terminated: bool


@dataclass(frozen=True, slots=True)
class TrainingResult:
    agent: QLearning
    episodes: tuple[Episode, ...]


def _positive(value: int) -> None:
    if type(value) is not int or value < 1:
        raise ValueError("episode count and step cap must be positive integers")


def _start(mdp: FiniteMDP, rng: Random) -> int:
    if mdp.initial is None:
        raise ValueError("environment requires an initial distribution")
    return rng.choices(range(mdp.n_states), weights=mdp.initial)[0]


def _episode(mdp: FiniteMDP, agent: QLearning, env_rng: Random,
             action_rng: Random, epsilon: float, cap: int, learn: bool) -> Episode:
    state = _start(mdp, env_rng)
    rewards = []
    for step in range(cap):
        if not mdp.actions(state):
            break
        action = agent.act(state, epsilon=epsilon, rng=action_rng)
        outcome = mdp.sample(state, action, env_rng)
        done = not mdp.actions(outcome.next_state)
        if learn:
            agent.update(state, action, outcome.reward, outcome.next_state,
                         terminated=done)
        rewards.append(mdp.gamma ** step * outcome.reward)
        state = outcome.next_state
        if done:
            break
    return Episode(fsum(rewards), len(rewards), not mdp.actions(state))


def train(mdp: FiniteMDP, *, episodes: int = 1000, cap: int = 100,
          seed: int = 0, alpha: float = 0.2, epsilon_start: float = 1.0,
          epsilon_end: float = 0.1) -> TrainingResult:
    """Linear exploration decay; the agent sees only sampled transitions."""
    _positive(episodes)
    _positive(cap)
    if not all(isfinite(e) for e in (epsilon_start, epsilon_end)) or not (
        0 <= epsilon_end <= epsilon_start <= 1
    ):
        raise ValueError("require 0 <= epsilon_end <= epsilon_start <= 1")
    agent = QLearning([len(mdp.actions(s)) for s in range(mdp.n_states)],
                      gamma=mdp.gamma, alpha=alpha)
    env_rng, action_rng = Random(seed), Random(seed + 1)
    history = []
    for i in range(episodes):
        fraction = i / max(1, episodes - 1)
        epsilon = epsilon_start + fraction * (epsilon_end - epsilon_start)
        history.append(_episode(mdp, agent, env_rng, action_rng, epsilon, cap, True))
    return TrainingResult(agent, tuple(history))


def evaluate(mdp: FiniteMDP, agent: QLearning, *, episodes: int = 100,
             cap: int = 100, seed: int = 10000,
             random_policy: bool = False) -> tuple[Episode, ...]:
    """Evaluate without learning; random baseline uses uniform legal actions."""
    _positive(episodes)
    _positive(cap)
    if agent.action_counts != tuple(len(mdp.actions(s)) for s in range(mdp.n_states)):
        raise ValueError("agent and environment action spaces differ")
    env_rng, action_rng = Random(seed), Random(seed + 1)
    return tuple(_episode(mdp, agent, env_rng, action_rng,
                          float(random_policy), cap, False) for _ in range(episodes))
