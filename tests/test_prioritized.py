from collections import Counter
from math import isclose

import pytest

from prometheus_rl.memory import PrioritizedReplayBuffer, Transition


def transition(i):
    return Transition((float(i),), 0, float(i), (0.0,), True)


def make_buffer(*, alpha=1.0, seed=9):
    buffer = PrioritizedReplayBuffer(5, 1, 1, alpha=alpha, seed=seed)
    for i in range(3):
        buffer.append(transition(i))
    buffer.update_priorities([0, 1, 2], [1, 2, 7])
    return buffer


def test_probabilities_and_weights():
    buffer = make_buffer()
    sample = buffer.sample_weighted(3, beta=1)
    masses = [1 + buffer.epsilon, 2 + buffer.epsilon, 7 + buffer.epsilon]
    assert buffer.total_mass == pytest.approx(sum(masses))
    for slot, prob, weight, t in zip(
        sample.indices, sample.probabilities, sample.weights,
        sample.transitions, strict=True,
    ):
        assert prob == pytest.approx(masses[slot] / sum(masses))
        assert weight == pytest.approx(masses[0] / masses[slot])
        assert t.reward == slot


def test_distribution_and_reproducibility():
    a, b = make_buffer(), make_buffer()
    draws = Counter()
    for _ in range(4000):
        x = a.sample_weighted(3, beta=0.5)
        assert x == b.sample_weighted(3, beta=0.5)
        draws.update(x.indices)
    for i, p in enumerate([0.1, 0.2, 0.7]):
        assert draws[i] / 12000 == pytest.approx(p, abs=0.02)


def test_uniform_alpha_and_beta_zero():
    buffer = make_buffer(alpha=0)
    sample = buffer.sample_weighted(3, beta=1)
    assert sample.probabilities == pytest.approx((1 / 3,) * 3)
    assert sample.weights == (1.0,) * 3
    assert make_buffer().sample_weighted(3, beta=0).weights == (1.0,) * 3


def test_overwrite_and_clear():
    buffer = make_buffer()
    for i in range(3, 6):
        buffer.append(transition(i))
    assert [t.reward for t in buffer.snapshot()] == [1, 2, 3, 4, 5]
    assert buffer.total_mass == pytest.approx(30 + 5 * buffer.epsilon)
    buffer.clear()
    assert len(buffer) == 0 and buffer.total_mass == 0
    buffer.append(transition(9))
    assert buffer.total_mass == 1
    assert buffer.sample_weighted(1, beta=1).weights == (1.0,)


def test_duplicate_and_zero_priority():
    buffer = make_buffer()
    buffer.update_priorities([0, 0, 1], [-4, 2, 0])
    assert buffer.total_mass == pytest.approx(11 + 3 * buffer.epsilon)
    assert buffer._min[1] == buffer.epsilon


@pytest.mark.parametrize("indices,errors", [
    ([0, 3], [2, 3]), ([0], []), ([0], [float('nan')]),
    ([-1], [1]), ([0], [float('inf')]),
])
def test_invalid_priority_updates_are_atomic(indices, errors):
    buffer = make_buffer()
    before = buffer.total_mass
    with pytest.raises(ValueError):
        buffer.update_priorities(indices, errors)
    assert buffer.total_mass == before


@pytest.mark.parametrize("size,beta", [(0, 1), (4, 1), (1, -1), (1, 2)])
def test_invalid_sampling(size, beta):
    with pytest.raises(ValueError):
        make_buffer().sample_weighted(size, beta=beta)


def test_single_slot():
    buffer = PrioritizedReplayBuffer(1, 1, 1)
    buffer.append(transition(1))
    buffer.update_priorities([0], [0])
    buffer.append(transition(2))
    assert isclose(buffer.total_mass, 1.0)
    assert buffer.sample(1) == (transition(2),)
