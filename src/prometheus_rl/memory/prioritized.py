"""Proportional prioritized replay with sum and minimum segment trees."""

from collections.abc import Sequence
from dataclasses import dataclass
from math import inf, isfinite, nextafter
from sys import float_info

from prometheus_rl.memory.replay import ReplayBuffer, Transition


@dataclass(frozen=True, slots=True)
class PrioritySample:
    transitions: tuple[Transition, ...]
    indices: tuple[int, ...]
    probabilities: tuple[float, ...]
    weights: tuple[float, ...]


class PrioritizedReplayBuffer(ReplayBuffer):
    """Single-threaded proportional replay, sampling with replacement.

    Slot indices are valid only until their experiences are overwritten.
    Update priorities immediately after sampling, before appending again.
    """

    def __init__(
        self, capacity: int, observation_dim: int, action_dim: int, *,
        alpha: float = 0.6, epsilon: float = 1e-6, seed: int = 0,
    ) -> None:
        super().__init__(capacity, observation_dim, action_dim, seed=seed)
        if not isfinite(alpha) or not 0 <= alpha <= 1:
            raise ValueError("alpha must be in [0, 1]")
        self._limit = float_info.max / (2 * capacity)
        if not isfinite(epsilon) or not 0 < epsilon <= self._limit:
            raise ValueError("epsilon must be positive and finite")
        self.alpha = alpha
        self.epsilon = epsilon
        self._base = 1 << (capacity - 1).bit_length()
        self._sum = [0.0] * (2 * self._base)
        self._min = [inf] * (2 * self._base)
        self._max_priority = max(1.0, epsilon)

    @property
    def total_mass(self) -> float:
        return self._sum[1]

    def _set(self, slot: int, priority: float) -> None:
        node = self._base + slot
        mass = priority ** self.alpha
        self._sum[node] = self._min[node] = mass
        while node > 1:
            node //= 2
            self._sum[node] = self._sum[2 * node] + self._sum[2 * node + 1]
            self._min[node] = min(self._min[2 * node], self._min[2 * node + 1])

    def append(self, transition: Transition) -> None:
        slot = self._next
        super().append(transition)
        self._set(slot, self._max_priority)

    def _draw(self) -> int:
        mass = min(self._rng.random() * self.total_mass,
                   nextafter(self.total_mass, 0.0))
        node = 1
        while node < self._base:
            node *= 2
            if mass >= self._sum[node]:
                mass -= self._sum[node]
                node += 1
            mass = min(mass, nextafter(self._sum[node], 0.0))
        return node - self._base

    def sample_weighted(self, batch_size: int, *, beta: float) -> PrioritySample:
        if type(batch_size) is not int or not 1 <= batch_size <= len(self):
            raise ValueError("batch size must be between one and buffer length")
        if not isfinite(beta) or not 0 <= beta <= 1:
            raise ValueError("beta must be in [0, 1]")
        slots = tuple(self._draw() for _ in range(batch_size))
        masses = tuple(self._sum[self._base + i] for i in slots)
        return PrioritySample(
            tuple(self._storage[i] for i in slots), slots,
            tuple(m / self.total_mass for m in masses),
            tuple((self._min[1] / m) ** beta if beta else 1.0 for m in masses),
        )

    def sample(self, batch_size: int) -> tuple[Transition, ...]:
        return self.sample_weighted(batch_size, beta=0.0).transitions

    def update_priorities(
        self, indices: Sequence[int], td_errors: Sequence[float],
    ) -> None:
        """Set priorities from absolute TD errors; duplicate slots use max error."""
        if len(indices) != len(td_errors):
            raise ValueError("indices and TD errors must match")
        updates: dict[int, float] = {}
        for i, error in zip(indices, td_errors, strict=True):
            if type(i) is not int or not 0 <= i < len(self):
                raise ValueError("invalid replay slot")
            priority = abs(float(error)) + self.epsilon
            if not isfinite(priority) or priority > self._limit:
                raise ValueError("TD error must yield a finite bounded priority")
            updates[i] = max(updates.get(i, 0.0), priority)
        for i, priority in updates.items():
            self._set(i, priority)
            self._max_priority = max(self._max_priority, priority)

    def clear(self) -> None:
        super().clear()
        self._sum = [0.0] * (2 * self._base)
        self._min = [inf] * (2 * self._base)
        self._max_priority = max(1.0, self.epsilon)
