"""Tabular off-policy TD control without a transition model."""

from collections.abc import Sequence
from math import isfinite
from random import Random


class QLearning:
    """Learn from sampled transitions with state-local action indices."""

    def __init__(self, action_counts: Sequence[int], *, gamma: float = 0.9,
                 alpha: float = 0.2) -> None:
        self.action_counts = tuple(action_counts)
        if not self.action_counts or any(
            type(n) is not int or n < 0 for n in self.action_counts
        ):
            raise ValueError("action counts must be nonnegative integers")
        if not isfinite(gamma) or not 0 <= gamma < 1:
            raise ValueError("gamma must be in [0, 1)")
        if not isfinite(alpha) or not 0 < alpha <= 1:
            raise ValueError("alpha must be in (0, 1]")
        self.gamma, self.alpha = gamma, alpha
        self._q = [[0.0] * n for n in self.action_counts]

    @property
    def values(self) -> tuple[tuple[float, ...], ...]:
        """Immutable snapshot of the learned action-value table."""
        return tuple(tuple(row) for row in self._q)

    def _row(self, state: int) -> list[float]:
        if type(state) is not int or not 0 <= state < len(self._q):
            raise ValueError("invalid state")
        return self._q[state]

    def greedy_action(self, state: int) -> int:
        row = self._row(state)
        if not row:
            raise ValueError("terminal state has no actions")
        return max(range(len(row)), key=row.__getitem__)

    def act(self, state: int, *, epsilon: float, rng: Random) -> int:
        """Epsilon-greedy with uniform exploration and deterministic greedy ties."""
        row = self._row(state)
        if not row or not isfinite(epsilon) or not 0 <= epsilon <= 1:
            raise ValueError("require legal actions and epsilon in [0, 1]")
        if rng.random() < epsilon:
            return rng.randrange(len(row))
        return self.greedy_action(state)

    def update(self, state: int, action: int, reward: float, next_state: int,
               *, terminated: bool) -> float:
        """Apply one TD update and return the pre-update TD error.

        Time-limit truncation is not termination and must not mask bootstrap.
        """
        row, successor = self._row(state), self._row(next_state)
        if type(action) is not int or not 0 <= action < len(row):
            raise ValueError("invalid action")
        if not isfinite(reward) or type(terminated) is not bool:
            raise ValueError("invalid reward or termination flag")
        if not terminated and not successor:
            raise ValueError("nonterminal successor must have legal actions")
        target = reward if terminated else reward + self.gamma * max(successor)
        error = target - row[action]
        updated = row[action] + self.alpha * error
        if not isfinite(error) or not isfinite(updated):
            raise ValueError("nonfinite TD update")
        row[action] = updated
        return error
