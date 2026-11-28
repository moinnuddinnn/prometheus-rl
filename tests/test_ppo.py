import math
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from prometheus_rl.agents.ppo import (  # noqa: E402
    PPOAgent,
    PPOConfig,
    Rollout,
    advantages,
    clipped_loss,
    discounted_returns,
    reinforce_loss,
)
from prometheus_rl.training.ppo import (  # noqa: E402
    evaluate,
    factory_for,
    train,
    train_reinforce,
)


@pytest.fixture(autouse=True)
def single_thread():
    original = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(original)


def test_gae_terminal_and_truncation():
    r = torch.tensor([1., 2., 3., 4.])
    v = torch.tensor([0., 0., 0., 0.])
    nv = torch.tensor([0., 10., 99., 10.])
    d = torch.tensor([False, False, True, False])
    u = torch.tensor([False, True, True, False])
    a, ret = advantages(r, v, nv, d, u, .9, .5)
    assert a.tolist() == pytest.approx([5.95, 11., 3., 13.])
    assert torch.equal(a, ret)


@pytest.mark.parametrize("ratio,adv,expected", [(1.5, 1., -1.2),
                         (.5, 1., -.5), (.5, -1., .8), (1.5, -1., 1.5)])
def test_clipping(ratio, adv, expected):
    logp = torch.tensor([math.log(ratio)], requires_grad=True)
    old = torch.zeros(1, requires_grad=True)
    a = torch.tensor([adv], requires_grad=True)
    loss = clipped_loss(logp, old, a, .2)
    assert loss.item() == pytest.approx(expected)
    loss.backward()
    assert old.grad is None and a.grad is None


def test_reinforce_gradient():
    logits = torch.zeros(2, requires_grad=True)
    returns = torch.tensor([2.], requires_grad=True)
    loss = reinforce_loss(logits.log_softmax(0)[1:], returns)
    loss.backward()
    assert logits.grad.tolist() == pytest.approx([1., -1.])
    assert returns.grad is None


@pytest.mark.parametrize("continuous", [False, True])
def test_actor_batch_and_update(continuous, tmp_path: Path):
    agent = PPOAgent(PPOConfig(3, 2, continuous=continuous), seed=12)
    x = torch.ones(9, 3)
    a, lp, val = agent.network.sample(x, agent.rng)
    assert a.shape == ((9, 2) if continuous else (9,))
    assert val.shape == lp.shape == (9,)
    current, entropy, _ = agent.network.statistics(x, a)
    assert torch.allclose(lp, current)
    assert torch.isfinite(entropy).all()
    if continuous:
        assert (a.tanh().abs() <= 1).all()
    old = [p.detach().clone() for p in agent.network.parameters()]
    metrics = agent.update(Rollout(x, a, lp, torch.arange(9).float(), val + 1))
    assert all(math.isfinite(v) for v in metrics.values())
    assert any(not torch.equal(a, b) for a, b in zip(
        old, agent.network.parameters(), strict=True))
    path = tmp_path / "model.pt"
    agent.save(path, game="test")
    loaded, game = PPOAgent.load(path)
    assert game == "test"
    assert loaded.updates == agent.updates
    assert torch.equal(loaded.network(x)[0], agent.network(x)[0])


class Bandit:
    def reset(self, *, seed=None):
        return (1.0,), {}

    def step(self, action):
        return (1.0,), float(action == 1), True, False, {}

    def close(self):
        pass


def test_online_bandit_learning_and_isolated_evaluation():
    cfg = PPOConfig(1, 2, learning_rate=.01, entropy_coef=0.)
    agent = PPOAgent(cfg, seed=2)
    episodes, metrics = train(agent, Bandit, steps=513, horizon=64, seed=2)
    assert len(episodes) == 513
    assert metrics[-1]["step"] == 513
    assert agent.network(torch.ones(1))[0].softmax(-1)[1] > .9
    rng = agent.rng.get_state().clone()
    before = agent.updates
    assert evaluate(agent, Bandit, episodes=20) == [1.] * 20
    assert sum(evaluate(agent, Bandit, episodes=100, random=True)) < 80
    assert agent.updates == before
    assert torch.equal(rng, agent.rng.get_state())


def test_ppo_gym_smoke():
    pytest.importorskip("gymnasium")
    from prometheus_rl.training.ppo import config_for

    for game in ("cartpole", "pendulum"):
        factory = factory_for(game)
        agent = PPOAgent(config_for(factory), seed=1)
        _, metrics = train(agent, factory, steps=32, horizon=16)
        assert len(metrics) == 2
        assert len(evaluate(agent, factory, episodes=1)) == 1


def test_monte_carlo_returns_and_reinforce_training():
    assert discounted_returns(torch.tensor([1., 2., 3.]), .5).tolist() == [
        2.75, 3.5, 3.
    ]
    agent = PPOAgent(PPOConfig(1, 2, learning_rate=.01), seed=2)
    scores = train_reinforce(agent, Bandit, episodes=100)
    assert len(scores) == agent.updates == 100
    assert evaluate(agent, Bandit, episodes=10) == [1.] * 10
