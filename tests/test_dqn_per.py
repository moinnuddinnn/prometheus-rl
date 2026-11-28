from dataclasses import replace

import pytest

torch = pytest.importorskip("torch")

from prometheus_rl.agents.dqn import DQNAgent, DQNConfig  # noqa: E402
from prometheus_rl.memory import PrioritizedReplayBuffer, Transition  # noqa: E402


def config(**kwargs):
    return DQNConfig(
        observation_dim=2, hidden_dims=(8,), capacity=8, batch_size=2,
        warmup=2, target_interval=2, prioritized=True, per_beta_steps=2,
        **kwargs,
    )


@pytest.mark.parametrize("double,dueling", [
    (False, False), (True, False), (False, True), (True, True),
])
def test_update_and_checkpoint(tmp_path, double, dueling):
    agent = DQNAgent(config(double=double, dueling=dueling))
    assert isinstance(agent.replay, PrioritizedReplayBuffer)
    assert agent.update() is None
    for reward in (2.0, 5.0):
        agent.observe(Transition((1., 0.), 0, reward, (0., 1.), True))
    initial = agent.replay.total_mass
    assert agent.replay_beta == pytest.approx(0.4)
    first = agent.update()
    assert first.loss > 0
    assert agent.replay.total_mass != initial
    assert agent.replay_beta == pytest.approx(0.7)
    assert agent.update().target_synced
    assert agent.replay_beta == 1
    path = tmp_path / "model.pt"
    agent.save(path)
    restored = DQNAgent.load(path)
    assert restored.config == agent.config
    assert restored.updates == 2 and restored.replay_beta == 1
    assert len(restored.replay) == 0
    assert torch.equal(restored.online(torch.ones(2)), agent.online(torch.ones(2)))


def test_weighted_huber_exact():
    agent = DQNAgent(config(per_alpha=1, per_beta_start=1))
    with torch.no_grad():
        for p in agent.online.parameters():
            p.zero_()
    for reward in (1., 4.):
        agent.observe(Transition((1., 0.), 0, reward, (0., 1.), True))
    agent.replay.update_priorities([0, 1], [1, 4])
    # The reference buffer uses the same seed as the agent's replay RNG.
    reference = PrioritizedReplayBuffer(8, 2, 2, alpha=1, seed=1)
    for t in agent.replay.snapshot():
        reference.append(t)
    reference.update_priorities([0, 1], [1, 4])
    sample = reference.sample_weighted(2, beta=1)
    expected = sum(w * (t.reward - 0.5) for w, t in zip(
        sample.weights, sample.transitions, strict=True,
    )) / 2
    assert agent.update().loss == pytest.approx(expected)


@pytest.mark.parametrize("kwargs", [
    {"per_alpha": -1}, {"per_beta_start": 2}, {"per_beta_steps": 0},
    {"per_epsilon": 0}, {"prioritized": 1},
])
def test_invalid_config(kwargs):
    with pytest.raises(ValueError):
        replace(config(), **kwargs)
