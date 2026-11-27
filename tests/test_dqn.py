from dataclasses import replace

import pytest

torch = pytest.importorskip("torch")

from prometheus_rl.agents.dqn import DQNAgent, DQNConfig, bellman_targets  # noqa: E402
from prometheus_rl.agents.dqn_demo import run_demo  # noqa: E402
from prometheus_rl.memory import Transition  # noqa: E402
from prometheus_rl.memory.batch import collate  # noqa: E402
from prometheus_rl.networks import QNetwork  # noqa: E402


@pytest.fixture
def cfg():
    return DQNConfig(observation_dim=2, hidden_dims=(), capacity=8,
                     batch_size=2, warmup=2, target_interval=2)


def transition(terminated=False, truncated=False):
    return Transition((1., 0.), 0, 2., (0., 1.), terminated, truncated)


@pytest.mark.parametrize("kwargs", [
    {"gamma": 1}, {"gamma": float("nan")}, {"batch_size": 3},
    {"warmup": 9}, {"target_interval": 0}, {"learning_rate": 0},
    {"max_grad_norm": float("inf")}, {"observation_dim": 0},
])
def test_invalid_config(cfg, kwargs):
    with pytest.raises(ValueError):
        replace(cfg, **kwargs)


def test_targets_and_flags():
    target = QNetwork(2, 2, ())
    with torch.no_grad():
        target.model[0].weight.zero_()
        target.model[0].bias.copy_(torch.tensor([3., 5.]))
    batch = collate([transition(), transition(True), transition(False, True),
                     transition(True, True)])
    y = bellman_targets(target, batch, .9)
    assert y.tolist() == pytest.approx([6.5, 2, 6.5, 2])
    assert not y.requires_grad
    assert torch.equal(bellman_targets(target, batch, 0), batch.rewards)


def test_warmup_and_target_sync(cfg):
    agent = DQNAgent(cfg)
    assert agent.update() is None
    agent.observe(transition(True))
    assert agent.update() is None
    agent.observe(transition(True))
    before = [p.clone() for p in agent.target.parameters()]
    result = agent.update()
    assert result.updates == 1 and not result.target_synced
    assert result.loss > 0 and result.grad_norm > 0
    assert all(torch.equal(p, old) for p, old in zip(
        agent.target.parameters(), before, strict=True))
    assert any(not torch.equal(p, old) for p, old in zip(
        agent.online.parameters(), before, strict=True))
    assert all(p.grad is None and not p.requires_grad
               for p in agent.target.parameters())
    assert agent.update().target_synced
    assert all(torch.equal(p, q) for p, q in zip(
        agent.online.parameters(), agent.target.parameters(), strict=True))


def test_seed_and_exploration(cfg):
    state = torch.random.get_rng_state().clone()
    a, b = DQNAgent(cfg, seed=42), DQNAgent(cfg, seed=42)
    assert torch.equal(state, torch.random.get_rng_state())
    assert all(torch.equal(p, q) for p, q in zip(
        a.online.parameters(), b.online.parameters(), strict=True))
    actions = [a.act((1, 0), epsilon=1) for _ in range(50)]
    assert actions == [b.act((1, 0), epsilon=1) for _ in range(50)]
    assert set(actions) == {0, 1}
    expected = int(a.online(torch.tensor([1., 0.])).argmax())
    assert a.act((1, 0)) == expected


@pytest.mark.parametrize("obs,epsilon", [([1], 0), ([1, float("nan")], 1),
                                         ([1, 0], -1), ([1, 0], 2)])
def test_bad_action_input(cfg, obs, epsilon):
    with pytest.raises(ValueError):
        DQNAgent(cfg).act(obs, epsilon=epsilon)


def test_gradient_clipping(cfg):
    a = DQNAgent(replace(cfg, max_grad_norm=.01))
    for _ in range(2):
        a.observe(Transition((100., 0.), 0, 100., (0., 1.), True))
    result = a.update()
    assert result.grad_norm > .01
    norm = torch.cat([p.grad.flatten() for p in a.online.parameters()]).norm()
    assert norm.item() <= .010001


def test_checkpoint(cfg, tmp_path):
    a = DQNAgent(cfg)
    for _ in range(2):
        a.observe(transition(True))
    a.update()
    path = tmp_path / "agent.pt"
    a.save(path)
    b = DQNAgent.load(path)
    assert b.config == a.config and b.updates == a.updates
    assert len(b.replay) == 0 and b.update() is None
    for net1, net2 in [(a.online, b.online), (a.target, b.target)]:
        assert all(torch.equal(p, q) for p, q in zip(
            net1.parameters(), net2.parameters(), strict=True))
    for _ in range(2):
        b.observe(transition(True))
    assert a.update().loss == pytest.approx(b.update().loss)
    assert all(torch.equal(p, q) for p, q in zip(
        a.online.parameters(), b.online.parameters(), strict=True))


def test_analytic_demo():
    q, v, action = run_demo()
    assert q == pytest.approx((2., 4.5), abs=.02)
    assert v == pytest.approx(5., abs=.02)
    assert action == 1
