"""Finite discounted MDPs with explicit stochastic transition outcomes."""

from collections.abc import Sequence
from dataclasses import dataclass
from math import fsum, isclose, isfinite
from random import Random


@dataclass(frozen=True, slots=True)
class Outcome:
    """One atom of the joint next-state/reward probability distribution."""

    probability: float
    next_state: int
    reward: float


@dataclass(frozen=True, slots=True)
class FiniteMDP:
    """Transitions indexed by state, action, outcome; empty states are terminal."""

    transitions: tuple[tuple[tuple[Outcome, ...], ...], ...]
    gamma: float = 0.9
    initial: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        rows = tuple(tuple(tuple(action) for action in row)
                     for row in self.transitions)
        object.__setattr__(self, "transitions", rows)
        if not rows or not isfinite(self.gamma) or not 0 <= self.gamma < 1:
            raise ValueError("require at least one state and 0 <= gamma < 1")
        for row in rows:
            for action in row:
                if not action:
                    raise ValueError("action must contain an outcome")
                for o in action:
                    if (not isfinite(o.probability) or o.probability < 0
                        or type(o.next_state) is not int
                        or not 0 <= o.next_state < len(rows)
                        or not isfinite(o.reward)):
                        raise ValueError("invalid transition outcome")
                _validate_distribution(tuple(o.probability for o in action))
        if self.initial is not None:
            initial = tuple(self.initial)
            if len(initial) != len(rows):
                raise ValueError("initial distribution must match state count")
            _validate_distribution(initial)
            object.__setattr__(self, "initial", initial)

    @property
    def n_states(self) -> int:
        return len(self.transitions)

    def actions(self, state: int) -> range:
        self._check_state(state)
        return range(len(self.transitions[state]))

    def _check_state(self, state: int) -> None:
        if type(state) is not int or not 0 <= state < self.n_states:
            raise ValueError("invalid state")

    def _outcomes(self, state: int, action: int) -> tuple[Outcome, ...]:
        if type(action) is not int or action not in self.actions(state):
            raise ValueError("invalid action")
        return self.transitions[state][action]

    def sample(self, state: int, action: int, rng: Random) -> Outcome:
        """Sample with a caller-owned random generator for reproducibility."""
        outcomes = self._outcomes(state, action)
        return rng.choices(outcomes, weights=[o.probability for o in outcomes])[0]

    def action_value(self, state: int, action: int, values: Sequence[float]) -> float:
        """Compute the expected reward plus discounted continuation."""
        return self._action_value(state, action, self.validate_values(values))

    def _action_value(self, state: int, action: int, values: Sequence[float]) -> float:
        outcomes = self._outcomes(state, action)
        return fsum(o.probability * (o.reward + self.gamma * values[o.next_state])
                    for o in outcomes)

    def validate_values(self, values: Sequence[float]) -> tuple[float, ...]:
        if len(values) != self.n_states or not all(isfinite(v) for v in values):
            raise ValueError("values must be finite and match state count")
        # Terminal states have zero continuation, even for arbitrary input vectors.
        return tuple(v if row else 0.0
                     for v, row in zip(values, self.transitions, strict=True))

    def expected_initial_value(self, values: Sequence[float]) -> float:
        if self.initial is None:
            raise ValueError("initial distribution not specified")
        return fsum(p * v for p, v in zip(
            self.initial, self.validate_values(values), strict=True))


def _validate_distribution(probabilities: Sequence[float]) -> None:
    if (not probabilities or any(not isfinite(p) or p < 0 for p in probabilities)
        or not isclose(fsum(probabilities), 1.0, rel_tol=0.0, abs_tol=1e-12)):
        raise ValueError("probabilities must be nonnegative and sum to one")
