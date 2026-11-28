from dataclasses import asdict

import pytest

torch = pytest.importorskip("torch")

from prometheus_rl.agents.dqn import DQNAgent, DQNConfig, bellman_targets  # noqa: E402
from prometheus_rl.memory import Transition  # noqa: E402
from prometheus_rl.memory.batch import collate  # noqa: E402
from prometheus_rl.networks import DuelingQNetwork, QNetwork  # noqa: E402


def test_double_selection_evaluation_and_flags():
    online, target = QNetwork(2, 2, ()), QNetwork(2, 2, ())
    with torch.no_grad():
        for net, bias in ((online, [9., 1.]), (target, [3., 5.])):
            net.model[0].weight.zero_()
            net.model[0].bias.copy_(torch.tensor(bias))
    batch = collate([
        Transition((1., 0.), 0, 2., (0., 1.), term, trunc)
        for term, trunc in ((False, False), (True, False), (False, True), (True, True))
    ])
    assert bellman_targets(target, batch, .9).tolist() == pytest.approx(
        [6.5, 2., 6.5, 2.])
    y = bellman_targets(target, batch, .9, online=online)
    assert y.tolist() == pytest.approx([4.7, 2., 4.7, 2.])
    assert not y.requires_grad


@pytest.mark.parametrize("shape", [(4,), (3, 4)])
def test_dueling_streams(shape):
    net = DuelingQNetwork(4, 2)
    x = torch.ones(shape)
    v, a = net.streams(x)
    q = net(x)
    assert q.shape == (*shape[:-1], 2)
    assert torch.allclose(q.mean(-1, keepdim=True), v, atol=1e-6)
    assert torch.allclose(q, v + a - a.mean(-1, keepdim=True))
    with torch.no_grad():
        net.advantage.bias.add_(5.)
    assert torch.allclose(net(x), q, atol=1e-6)


def test_dueling_gradients():
    net = DuelingQNetwork(2, 2, ())
    net(torch.tensor([1., 1.]))[0].backward()
    assert net.value.bias.grad.tolist() == [1.]
    assert net.advantage.bias.grad.tolist() == [.5, -.5]


@pytest.mark.parametrize("double", [False, True])
@pytest.mark.parametrize("dueling", [False, True])
def test_variants_train_and_checkpoint(double, dueling, tmp_path):
    cfg = DQNConfig(observation_dim=2, hidden_dims=(8,), capacity=8,
                    batch_size=2, warmup=2, target_interval=2,
                    double=double, dueling=dueling)
    agent = DQNAgent(cfg, seed=12)
    assert isinstance(agent.online, DuelingQNetwork if dueling else QNetwork)
    for i in range(2):
        agent.observe(Transition((1., 0.), i, float(i), (0., 1.), False))
    assert not agent.update().target_synced
    assert all(p.grad is None for p in agent.target.parameters())
    assert agent.update().target_synced
    path = tmp_path / "model.pt"
    agent.save(path)
    restored = DQNAgent.load(path)
    assert restored.config == cfg
    assert restored.updates == 2
    assert torch.equal(agent.online(torch.ones(2)), restored.online(torch.ones(2)))


def test_legacy_checkpoint(tmp_path):
    agent = DQNAgent(DQNConfig())
    config = asdict(agent.config)
    del config["double"], config["dueling"]
    path = tmp_path / "old.pt"
    torch.save({"format": 1, "config": config, "online": agent.online.state_dict(),
                "target": agent.target.state_dict(),
                "optimizer": agent.optimizer.state_dict(), "updates": 0}, path)
    restored = DQNAgent.load(path)
    assert not restored.config.double and not restored.config.dueling
    assert torch.equal(agent.online(torch.ones(4)), restored.online(torch.ones(4)))


@pytest.mark.parametrize("kwargs", [{"double": 1}, {"dueling": "yes"}])
def test_invalid_variant(kwargs):
    with pytest.raises(ValueError):
        DQNConfig(**kwargs)


def test_cli_flags(monkeypatch, tmp_path):
    from prometheus_rl.training import cartpole_cli

    def experiment(*args, **kwargs):
        assert kwargs["dqn_config"].double and kwargs["dqn_config"].dueling
        return {"greedy": {"mean": 1.}, "random": {"mean": 0.}}

    monkeypatch.setattr(cartpole_cli, "run_experiment", experiment)
    assert cartpole_cli.main(["train", "--double", "--dueling", "--output",
                              str(tmp_path)]) == 0


@pytest.mark.parametrize("double,dueling", [(True, False), (False, True), (True, True)])
def test_variant_analytic_solution(double, dueling):
    from prometheus_rl.agents.dqn_demo import run_demo

    q, v, action = run_demo(double=double, dueling=dueling)
    assert q == pytest.approx((2., 4.5), abs=.02)
    assert v == pytest.approx(5., abs=.02)
    assert action == 1
