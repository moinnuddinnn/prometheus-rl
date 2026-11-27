from collections import Counter
from dataclasses import FrozenInstanceError

import pytest

from prometheus_rl.memory import ReplayBuffer, Transition


def transition(i):
    return Transition((float(i),), i % 2, float(i), (float(i + 1),), False)


def test_ring_fifo_and_full_sample():
    buffer = ReplayBuffer(3, 1, 2)
    for i in range(12):
        buffer.append(transition(i))
        assert len(buffer) == min(i + 1, 3)
        assert [t.reward for t in buffer.snapshot()] == list(
            range(max(0, i - 2), i + 1))
    assert set(buffer.sample(3)) == set(buffer.snapshot())


def test_capacity_one_clear_and_refill():
    buffer = ReplayBuffer(1, 1, 2)
    buffer.append(transition(0))
    buffer.append(transition(1))
    assert buffer.sample(1) == (transition(1),)
    buffer.clear()
    assert not buffer.snapshot()
    buffer.append(transition(2))
    assert buffer.snapshot() == (transition(2),)


def test_seeded_sampling_and_distinct_slots():
    a, b = (ReplayBuffer(10, 1, 2, seed=42) for _ in range(2))
    for i in range(15):
        for buffer in (a, b):
            buffer.append(transition(i))
    for _ in range(20):
        x, y = a.sample(4), b.sample(4)
        assert x == y
        assert len(set(x)) == 4


def test_uniform_inclusion_smoke():
    buffer = ReplayBuffer(5, 1, 2, seed=19)
    for i in range(5):
        buffer.append(transition(i))
    counts = Counter(t.reward for _ in range(5000) for t in buffer.sample(2))
    assert all(abs(counts[i] - 2000) < 200 for i in range(5))


def test_immutable_snapshot_of_mutable_observations():
    source = [1.0]
    t = Transition(source, 0, 0.0, source, False)
    source[0] = 999.0
    assert t.observation == t.next_observation == (1.0,)
    with pytest.raises(FrozenInstanceError):
        t.reward = 5


@pytest.mark.parametrize("terminated,truncated,mask,end", [
    (False, False, 1, False), (False, True, 1, True),
    (True, False, 0, True), (True, True, 0, True),
])
def test_flags(terminated, truncated, mask, end):
    t = Transition((0,), 0, 1, (1,), terminated, truncated)
    assert t.bootstrap_mask == mask
    assert t.episode_end is end


@pytest.mark.parametrize("args", [(0, 1, 2), (2, 0, 2), (2, 1, True)])
def test_invalid_buffer(args):
    with pytest.raises(ValueError):
        ReplayBuffer(*args)


@pytest.mark.parametrize("size", [0, -1, 2, True])
def test_invalid_sample(size):
    buffer = ReplayBuffer(2, 1, 2)
    buffer.append(transition(0))
    with pytest.raises(ValueError):
        buffer.sample(size)


@pytest.mark.parametrize("args", [
    ((), 0, 0, (), False), ((0,), -1, 0, (1,), False),
    ((0,), 0, float("nan"), (1,), False), ((0,), 0, 0, (1, 2), False),
    ((0,), 0, 0, (1,), 1), ((float("inf"),), 0, 0, (1,), False),
])
def test_invalid_transition(args):
    with pytest.raises(ValueError):
        Transition(*args)


def test_reject_mismatched_transitions_without_mutation():
    buffer = ReplayBuffer(2, 1, 2)
    buffer.append(transition(0))
    for t in (Transition((0, 1), 0, 0, (1, 2), False),
              Transition((0,), 2, 0, (1,), False)):
        with pytest.raises(ValueError):
            buffer.append(t)
    assert buffer.snapshot() == (transition(0),)
