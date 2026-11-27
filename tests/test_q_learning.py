from random import Random
from statistics import mean

import pytest

from prometheus_rl.agents import QLearning
from prometheus_rl.core import value_iteration
from prometheus_rl.environments.gridworld import gridworld
from prometheus_rl.training.tabular import evaluate, train


def test_hand_calculated_updates():
    agent = QLearning((2, 1, 0), alpha=0.5, gamma=0.9)
    assert agent.update(1, 0, 5, 2, terminated=True) == 5
    assert agent.update(0, 1, 0, 1, terminated=False) == 2.25
    assert agent.values == ((0.0, 1.125), (2.5,), ())


def test_termination_masks_nonzero_bootstrap():
    agent = QLearning((1, 1), alpha=1)
    agent.update(1, 0, 100, 1, terminated=True)
    agent.update(0, 0, 2, 1, terminated=True)
    assert agent.values[0] == (2.0,)
    agent.update(0, 0, 2, 1, terminated=False)
    assert agent.values[0] == (92.0,)


def test_exploration_and_greedy_ties():
    agent = QLearning((4, 0))
    rng = Random(1)
    assert agent.act(0, epsilon=0, rng=rng) == 0
    assert {agent.act(0, epsilon=1, rng=rng) for _ in range(100)} == set(range(4))
    with pytest.raises(ValueError):
        agent.act(1, epsilon=0, rng=rng)


@pytest.mark.parametrize(
    "kwargs", [{"gamma": 1}, {"alpha": 0}, {"alpha": float("nan")}],
)
def test_invalid_hyperparameters(kwargs):
    with pytest.raises(ValueError):
        QLearning((1,), **kwargs)


def test_invalid_transition():
    agent = QLearning((1, 0))
    with pytest.raises(ValueError):
        agent.update(0, 0, 0, 1, terminated=False)
    with pytest.raises(ValueError):
        agent.update(0, -1, 0, 1, terminated=True)
    assert agent.values == ((0.0,), ())


def test_seeded_training_and_frozen_evaluation():
    mdp = gridworld()
    a, b = train(mdp, seed=3), train(mdp, seed=3)
    assert a.episodes == b.episodes
    assert a.agent.values == b.agent.values
    before = a.agent.values
    assert evaluate(mdp, a.agent) == evaluate(mdp, a.agent)
    assert a.agent.values == before


@pytest.mark.parametrize("seed", range(5))
def test_gridworld_optimal_policy(seed):
    mdp = gridworld()
    trained = train(mdp, seed=seed)
    greedy = evaluate(mdp, trained.agent, seed=10000 + seed * 2)
    random = evaluate(mdp, trained.agent, seed=10000 + seed * 2, random_policy=True)
    optimal = value_iteration(mdp).values[0]
    assert all(e.terminated and e.steps == 6 for e in greedy)
    assert mean(e.discounted_return for e in greedy) == pytest.approx(optimal)
    assert mean(e.discounted_return for e in greedy) > mean(
        e.discounted_return for e in random)


def test_cap_is_not_terminal():
    mdp = gridworld()
    trained = train(mdp, episodes=2, cap=1)
    assert all(e.steps == 1 and not e.terminated for e in trained.episodes)


def test_cli_demo(capsys):
    from prometheus_rl.__main__ import main

    assert main(["q-demo"]) == 0
    assert "success=100%" in capsys.readouterr().out
