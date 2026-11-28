"""Synthetic image contracts and optional actual ALE smoke test."""

import importlib.util

import pytest

if any(importlib.util.find_spec(m) is None for m in ("torch", "numpy")):
    pytest.skip("visual dependencies unavailable", allow_module_level=True)

import numpy as np  # noqa: E402
import torch  # noqa: E402

from prometheus_rl.agents.visual import VisualAgent, VisualConfig  # noqa: E402
from prometheus_rl.memory.pixels import PixelExperience, PixelReplay  # noqa: E402
from prometheus_rl.networks.visual import VisualQNetwork  # noqa: E402
from prometheus_rl.training.atari import evaluate, train  # noqa: E402
from prometheus_rl.training.cartpole import TrainConfig  # noqa: E402


def pixels(value=0):
    return np.full((4, 84, 84), value, dtype=np.uint8)


@pytest.mark.parametrize("dueling", [False, True])
def test_cnn_shape_and_gradients(dueling):
    model = VisualQNetwork(4, dueling=dueling)
    x = torch.full((2, 4, 84, 84), 128, dtype=torch.uint8)
    y = model(x)
    assert y.shape == (2, 4)
    y.square().sum().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all()
               for p in model.parameters())
    if dueling:
        h = model.features(x.float() / 255)
        assert torch.allclose(y.mean(1), model.value(h).squeeze(1), atol=1e-6)


def test_cnn_invalid_inputs():
    model = VisualQNetwork(2)
    with pytest.raises(ValueError, match="uint8"):
        model(torch.zeros(1, 4, 84, 84))
    with pytest.raises(ValueError, match="expected"):
        model(torch.zeros(4, 84, 84, dtype=torch.uint8))


def test_pixel_replay_copy_eviction_and_flags():
    replay = PixelReplay(2, 2, seed=7)
    x = pixels(3)
    replay.append(PixelExperience(x, 0, 1, x, False, True))
    x[:] = 255
    assert replay.sample(1).observations.max() == 3
    replay.append(PixelExperience(pixels(4), 1, 2, pixels(5), True, False))
    replay.append(PixelExperience(pixels(6), 0, 3, pixels(7), False, True))
    batch = replay.sample(2)
    assert sorted(batch.rewards.tolist()) == [2, 3]
    assert sorted(batch.bootstrap_mask.tolist()) == [0, 1]
    assert batch.episode_end.all()
    assert batch.observations.dtype == torch.uint8
    assert replay.pixel_bytes == 2 * 2 * 4 * 84 * 84


def test_invalid_pixel_transition():
    replay = PixelReplay(2, 2)
    with pytest.raises(ValueError, match="uint8"):
        replay.append(PixelExperience(pixels().astype(float), 0, 0,
                                      pixels(), False, False))
    with pytest.raises(ValueError, match="invalid action"):
        replay.append(PixelExperience(pixels(), 2, 0, pixels(), False, False))
    assert len(replay) == 0


@pytest.mark.parametrize("double,dueling", [(False, False), (True, True)])
def test_agent_update_and_checkpoint(tmp_path, double, dueling):
    cfg = VisualConfig(2, capacity=4, batch_size=2, warmup=2,
                       target_interval=2, double=double, dueling=dueling)
    agent = VisualAgent(cfg, seed=4)
    assert agent.update() is None
    for action in range(2):
        agent.observe(PixelExperience(pixels(50), action, 1, pixels(60),
                                      bool(action), False))
    before = [p.clone() for p in agent.target.parameters()]
    first = agent.update()
    assert first is not None and not first.target_synced
    assert all(torch.equal(a, b) for a, b in zip(
        before, agent.target.parameters(), strict=True))
    second = agent.update()
    assert second is not None and second.target_synced
    assert all(p.grad is None for p in agent.target.parameters())
    assert all(torch.equal(a, b) for a, b in zip(
        agent.online.parameters(), agent.target.parameters(), strict=True))
    path = tmp_path / "visual.pt"
    agent.save(path, game="pong")
    restored, game = VisualAgent.load(path)
    assert game == "pong" and restored.updates == 2 and len(restored.replay) == 0
    assert restored.act(pixels(50)) == agent.act(pixels(50))


class TinyPixels:
    def reset(self, *, seed=None):
        return pixels(100), {}

    def step(self, action):
        return pixels(0), 5.0 if action == 1 else -2.0, True, False, {}

    def close(self):
        pass


def test_online_pixel_loop_and_raw_scores():
    agent = VisualAgent(VisualConfig(2, capacity=8, batch_size=2, warmup=2), seed=2)
    records = train(agent, TrainConfig(steps=8, train_every=2, seed=2), TinyPixels)
    assert len(records) == 8 and agent.updates == 4
    assert all(r.reward in (-2, 5) for r in records)
    assert set(agent.replay.sample(8).rewards.tolist()) <= {-1, 1}
    before = agent.updates
    scores = evaluate(agent, TinyPixels, episodes=3, seed=100, max_steps=1)
    assert len(scores) == 3 and all(s in (-2, 5) for s in scores)
    assert agent.updates == before and len(agent.replay) == 8


@pytest.mark.parametrize("game", ["breakout", "pong"])
def test_real_atari_smoke(game):
    pytest.importorskip("gymnasium")
    pytest.importorskip("ale_py")
    pytest.importorskip("cv2")
    from prometheus_rl.environments.atari import make_atari

    env = make_atari(game)
    try:
        obs, _ = env.reset(seed=1)
        assert obs.shape == (4, 84, 84) and obs.dtype == np.uint8
        nxt, _, _, _, _ = env.step(0)
        assert nxt.shape == obs.shape and nxt.dtype == np.uint8
    finally:
        env.close()
