"""Synchronous dynamic programming with Bellman residual error bounds."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from math import fsum, isfinite

from prometheus_rl.core.mdp import FiniteMDP, _validate_distribution

Policy = tuple[tuple[float, ...], ...]


@dataclass(frozen=True, slots=True)
class PlanningResult:
    values: tuple[float, ...]
    policy: Policy
    iterations: int
    residual: float
    error_bound: float


def bellman_optimality(mdp: FiniteMDP, values: Sequence[float]) -> tuple[float, ...]:
    v = mdp.validate_values(values)
    return tuple(max((mdp._action_value(s, a, v) for a in mdp.actions(s)), default=0.0)
                 for s in range(mdp.n_states))


def _validate_policy(mdp: FiniteMDP, policy: Policy) -> Policy:
    if len(policy) != mdp.n_states:
        raise ValueError("policy must match state count")
    rows = tuple(tuple(row) for row in policy)
    for s, row in enumerate(rows):
        if len(row) != len(mdp.actions(s)):
            raise ValueError("policy must match legal actions")
        if row:
            _validate_distribution(row)
    return rows


def _policy_operator(mdp: FiniteMDP, policy: Policy,
                     values: Sequence[float]) -> tuple[float, ...]:
    v = mdp.validate_values(values)
    return tuple(fsum(p * mdp._action_value(s, a, v) for a, p in enumerate(row))
                 for s, row in enumerate(policy))


def bellman_policy(mdp: FiniteMDP, policy: Policy,
                   values: Sequence[float]) -> tuple[float, ...]:
    return _policy_operator(mdp, _validate_policy(mdp, policy), values)


def greedy_policy(mdp: FiniteMDP, values: Sequence[float]) -> Policy:
    """Choose the lowest-index maximizing action; terminal rows are empty."""
    v = mdp.validate_values(values)
    rows = []
    for s in range(mdp.n_states):
        actions = mdp.actions(s)
        if actions:
            best = max(actions, key=lambda a: mdp._action_value(s, a, v))
            rows.append(tuple(float(a == best) for a in actions))
        else:
            rows.append(())
    return tuple(rows)


def _solve(
    mdp: FiniteMDP,
    operator: Callable[[tuple[float, ...]], tuple[float, ...]],
    tolerance: float,
    max_iterations: int,
) -> tuple[tuple[float, ...], int, float]:
    if not isfinite(tolerance) or tolerance <= 0:
        raise ValueError("tolerance must be positive and finite")
    if type(max_iterations) is not int or max_iterations < 1:
        raise ValueError("max_iterations must be a positive integer")
    values = (0.0,) * mdp.n_states
    for iteration in range(max_iterations + 1):
        updated = operator(values)
        residual = max(abs(a - b) for a, b in zip(updated, values, strict=True))
        if residual / (1 - mdp.gamma) <= tolerance:
            return values, iteration, residual
        values = updated
    raise RuntimeError("Bellman error tolerance not reached within iteration budget")


def value_iteration(mdp: FiniteMDP, *, tolerance: float = 1e-9,
                    max_iterations: int = 100_000) -> PlanningResult:
    """Approximate V* with a certified infinity-norm error in exact arithmetic."""
    v, n, r = _solve(mdp, lambda x: bellman_optimality(mdp, x),
                     tolerance, max_iterations)
    return PlanningResult(v, greedy_policy(mdp, v), n, r, r / (1 - mdp.gamma))


def evaluate_policy(mdp: FiniteMDP, policy: Policy, *, tolerance: float = 1e-9,
                    max_iterations: int = 100_000) -> PlanningResult:
    """Evaluate a stationary stochastic policy by fixed-point iteration."""
    policy = _validate_policy(mdp, policy)
    v, n, r = _solve(mdp, lambda x: _policy_operator(mdp, policy, x),
                     tolerance, max_iterations)
    return PlanningResult(v, policy, n, r, r / (1 - mdp.gamma))
