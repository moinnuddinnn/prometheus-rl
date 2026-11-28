"""Online control contracts; Gymnasium integration is tested when installed."""

import json
from pathlib import Path

import pytest

pytest.importorskip("torch")
import torch  # noqa: E402

from prometheus_rl.agents.dqn import DQNAgent, DQNConfig  # noqa: E402
from prometheus_rl.training.cartpole import (  # noqa: E402
    TrainConfig,
    evaluate,
    plot_results,
    run_experiment,
    train,
    write_results,
)


class TinyEnv:
    """A single-step two-action task, with alternating end flags."""

    def __init__(self):
        self.seeds = []
        self.closed = False

    def reset(self, *, seed):
        self.seeds.append(seed)
        return [1.0, 0.0, 0.0, 0.0], {}

    def step(self, action):
        terminal = len(self.seeds) % 2 == 1
        return [0.0] * 4, float(action), terminal, not terminal, {}

    def close(self):
        self.closed = True


def agent():
    return DQNAgent(DQNConfig(hidden_dims=(16,), capacity=256, batch_size=8,
                             warmup=8, target_interval=10), seed=4)


def test_schedule():
    cfg = TrainConfig(decay_steps=10)
    assert cfg.epsilon(0) == 1.0
    assert cfg.epsilon(5) == pytest.approx(0.525)
    assert cfg.epsilon(20) == pytest.approx(0.05)


@pytest.mark.parametrize("kwargs", [dict(steps=0), dict(train_every=0),
    dict(decay_steps=0), dict(seed=-1), dict(epsilon_start=0.1, epsilon_end=0.2)])
def test_invalid_config(kwargs):
    with pytest.raises(ValueError):
        TrainConfig(**kwargs)


def test_online_collection_and_flags():
    model = agent()
    env = TinyEnv()
    callbacks = []
    records = train(model, TrainConfig(steps=20, train_every=4, seed=7),
                    factory=lambda: env, on_episode=callbacks.append)
    assert len(records) == len(model.replay) == 20
    assert model.updates == 4
    assert tuple(callbacks) == records
    assert env.seeds == [7] + [None] * 19
    assert env.closed
    assert records[0].terminated and records[1].truncated
    assert records[0].mean_loss is None
    assert records[7].mean_loss is not None
    assert model.replay.snapshot()[1].bootstrap_mask == 1.0


def test_online_learning_and_frozen_evaluation():
    model = agent()
    train(model, TrainConfig(steps=800, train_every=1, decay_steps=500),
          factory=TinyEnv)
    before = {k: v.clone() for k, v in model.online.state_dict().items()}
    updates = model.updates
    replay = model.replay.snapshot()
    env = TinyEnv()
    result = evaluate(model, episodes=50, seed=1000, factory=lambda: env)
    random = evaluate(model, episodes=50, seed=1000, random_policy=True,
                      factory=TinyEnv)
    assert result.average == 1.0 > random.average
    assert env.seeds == list(range(1000, 1050)) and env.closed
    assert model.updates == updates and model.replay.snapshot() == replay
    assert all(torch.equal(v, model.online.state_dict()[k]) for k, v in before.items())


def test_logs_and_plots(tmp_path: Path):
    pytest.importorskip("matplotlib")
    records = train(agent(), TrainConfig(steps=12, train_every=1), factory=TinyEnv)
    write_results(tmp_path, records, {"test": True})
    plot_results(tmp_path, records)
    assert len((tmp_path / "episodes.csv").read_text().splitlines()) == 13
    assert json.loads((tmp_path / "summary.json").read_text()) == {"test": True}
    assert (tmp_path / "returns.png").stat().st_size > 0
    assert (tmp_path / "loss.png").stat().st_size > 0


def test_reject_overlapping_seeds(tmp_path):
    with pytest.raises(ValueError, match="overlap"):
        run_experiment(tmp_path / "run", TrainConfig(steps=10), eval_seed=5)


def test_real_gymnasium_smoke(tmp_path):
    pytest.importorskip("gymnasium")
    path = tmp_path / "cartpole"
    result = run_experiment(path, TrainConfig(steps=128, train_every=4),
                            eval_episodes=2, plots=False,
                            dqn_config=agent().config)
    assert result["updates"] > 0
    assert 1 <= result["greedy"]["mean"] <= 500
    assert len(result["random"]["returns"]) == 2
    restored = DQNAgent.load(path / "model.pt")
    assert restored.updates == result["updates"]
    assert evaluate(restored, episodes=2).returns == tuple(result["greedy"]["returns"])
