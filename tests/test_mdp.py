from random import Random

import pytest

from prometheus_rl.__main__ import main
from prometheus_rl.core import (
    FiniteMDP,
    Outcome,
    bellman_optimality,
    bellman_policy,
    evaluate_policy,
    greedy_policy,
    value_iteration,
)
from prometheus_rl.core.examples import delayed_reward


def test_delayed_reward_exact():
    mdp = delayed_reward()
    result = value_iteration(mdp)
    assert result.values == pytest.approx((4.5, 5.0, 0.0))
    assert result.policy == ((0.0, 1.0), (1.0,), ())
    assert result.residual == 0
    assert mdp.expected_initial_value(result.values) == pytest.approx(4.5)


def test_policy_evaluation():
    mdp = delayed_reward()
    result = evaluate_policy(mdp, ((0.5, 0.5), (1.0,), ()))
    assert result.values == pytest.approx((3.25, 5, 0))
    assert bellman_policy(mdp, result.policy, result.values) == result.values


def test_geometric_loop_error_bound():
    mdp = FiniteMDP((((Outcome(1, 0, 1),),),), gamma=0.9)
    result = value_iteration(mdp, tolerance=1e-7)
    assert abs(result.values[0] - 10) <= result.error_bound + 1e-12
    assert result.error_bound <= 1e-7


def test_stochastic_joint_outcomes_and_terminal_mask():
    mdp = FiniteMDP((((Outcome(0.25, 1, 8), Outcome(0.75, 0, 0)),), ()),
                    gamma=0.5)
    assert mdp.action_value(0, 0, (4, 999)) == pytest.approx(3.5)
    result = value_iteration(mdp)
    assert result.values == pytest.approx((3.2, 0))
    a, b = Random(42), Random(42)
    assert [mdp.sample(0, 0, a) for _ in range(30)] == [
        mdp.sample(0, 0, b) for _ in range(30)]


@pytest.mark.parametrize("outcomes", [(), (Outcome(0.8, 0, 1),),
    (Outcome(-1, 0, 1), Outcome(2, 0, 1)), (Outcome(1, 2, 0),),
    (Outcome(1, 0, float("nan")),), (Outcome(float("inf"), 0, 1),)])
def test_invalid_outcomes(outcomes):
    with pytest.raises(ValueError):
        FiniteMDP(((outcomes,),))


@pytest.mark.parametrize("gamma", [-1, 1, float("nan")])
def test_invalid_discount(gamma):
    with pytest.raises(ValueError):
        FiniteMDP(((),), gamma=gamma)


def test_invalid_state_action_policy_values():
    mdp = delayed_reward()
    for s, a in [(-1, 0), (0, -1), (2, 0), (0, 2)]:
        with pytest.raises(ValueError):
            mdp.action_value(s, a, (0, 0, 0))
    with pytest.raises(ValueError):
        evaluate_policy(mdp, ((1, 1), (1,), ()))
    with pytest.raises(ValueError):
        bellman_optimality(mdp, (0, 1))
    with pytest.raises(ValueError):
        FiniteMDP(((),), initial=(0.5,))


def test_contraction():
    mdp = delayed_reward()
    v, w = (1, 3, 0), (2, -1, 0)
    tv, tw = bellman_optimality(mdp, v), bellman_optimality(mdp, w)
    assert max(abs(a-b) for a, b in zip(tv, tw, strict=True)) <= mdp.gamma * 4


def test_ties_and_zero_discount():
    mdp = FiniteMDP((((Outcome(1, 0, 2),), (Outcome(1, 0, 2),)),), gamma=0)
    assert greedy_policy(mdp, (0,)) == ((1.0, 0.0),)
    assert value_iteration(mdp).values == (2,)


def test_nonconvergence():
    mdp = FiniteMDP((((Outcome(1, 0, 1),),),), gamma=0.99)
    with pytest.raises(RuntimeError):
        value_iteration(mdp, max_iterations=2)


def test_demo(capsys):
    assert main(["mdp-demo"]) == 0
    assert "Initial expected return: 4.50" in capsys.readouterr().out
