"""Uniform experience replay with fixed-capacity FIFO eviction."""

from collections.abc import Sequence
from dataclasses import dataclass
from math import isfinite
from random import Random


@dataclass(frozen=True, slots=True)
class Transition:
    """Immutable snapshot of one environment interaction."""

    observation: tuple[float, ...]
    action: int
    reward: float
    next_observation: tuple[float, ...]
    terminated: bool
    truncated: bool = False

    def __post_init__(self) -> None:
        obs = tuple(float(x) for x in self.observation)
        nxt = tuple(float(x) for x in self.next_observation)
        reward = float(self.reward)
        if not obs or len(obs) != len(nxt):
            raise ValueError("observations must have matching nonzero dimensions")
        if not all(isfinite(x) for x in (*obs, *nxt, reward)):
            raise ValueError("observations and reward must be finite")
        if type(self.action) is not int or self.action < 0:
            raise ValueError("action must be a nonnegative integer")
        if type(self.terminated) is not bool or type(self.truncated) is not bool:
            raise ValueError("episode flags must be booleans")
        object.__setattr__(self, "observation", obs)
        object.__setattr__(self, "next_observation", nxt)
        object.__setattr__(self, "reward", reward)

    @property
    def episode_end(self) -> bool:
        return self.terminated or self.truncated

    @property
    def bootstrap_mask(self) -> float:
        return float(not self.terminated)


class ReplayBuffer:
    """Single-threaded ring buffer; sample distinct occupied slots uniformly."""

    def __init__(
        self, capacity: int, observation_dim: int, action_dim: int, *,
        seed: int = 0,
    ) -> None:
        if any(type(x) is not int or x <= 0
               for x in (capacity, observation_dim, action_dim)):
            raise ValueError("capacity and dimensions must be positive integers")
        self.capacity = capacity
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self._storage: list[Transition] = []
        self._next = 0
        self._rng = Random(seed)

    def __len__(self) -> int:
        return len(self._storage)

    def append(self, transition: Transition) -> None:
        if not isinstance(transition, Transition):
            raise TypeError("expected a Transition")
        if len(transition.observation) != self.observation_dim:
            raise ValueError("observation dimension does not match buffer")
        if transition.action >= self.action_dim:
            raise ValueError("action out of bounds")
        if len(self) < self.capacity:
            self._storage.append(transition)
        else:
            self._storage[self._next] = transition
        self._next = (self._next + 1) % self.capacity

    def snapshot(self) -> tuple[Transition, ...]:
        """Return the retained experiences in oldest-to-newest order."""
        start = self._next if len(self) == self.capacity else 0
        return tuple(self._storage[start:] + self._storage[:start])

    def sample(self, batch_size: int) -> tuple[Transition, ...]:
        if type(batch_size) is not int or not 1 <= batch_size <= len(self):
            raise ValueError("batch size must be between one and buffer length")
        return tuple(self._rng.sample(self._storage, batch_size))

    def clear(self) -> None:
        """Remove stored experiences; preserve the random generator state."""
        self._storage.clear()
        self._next = 0


def replay_demo() -> tuple[Sequence[float], Sequence[float], Sequence[float]]:
    buffer = ReplayBuffer(4, 1, 2, seed=7)
    for i in range(6):
        buffer.append(Transition((float(i),), i % 2, float(i),
                                 (float(i + 1),), i == 4, i == 5))
    batch = buffer.sample(3)
    return ([t.reward for t in buffer.snapshot()], [t.reward for t in batch],
            [t.bootstrap_mask for t in batch])
