import pytest

from prometheus_rl.memory import Transition


def test_tensor_contract_and_network_compatibility():
    torch = pytest.importorskip("torch")
    from prometheus_rl.memory.batch import collate
    from prometheus_rl.networks import QNetwork, select_q

    transitions = [Transition((0, 1), 0, 2, (1, 2), False, True),
                   Transition((2, 3), 1, -1, (3, 4), True)]
    batch = collate(transitions)
    assert batch.observations.shape == batch.next_observations.shape == (2, 2)
    assert batch.actions.dtype == torch.long
    assert batch.rewards.dtype == torch.float32
    assert batch.terminated.dtype == batch.truncated.dtype == torch.bool
    assert batch.actions.shape == batch.rewards.shape == (2,)
    assert batch.bootstrap_mask.tolist() == [1, 0]
    assert batch.episode_end.tolist() == [True, True]
    assert not batch.observations.requires_grad
    assert select_q(QNetwork(2, 2)(batch.observations), batch.actions).shape == (2,)


def test_batch_singleton_double_and_invalid_cases():
    torch = pytest.importorskip("torch")
    from prometheus_rl.memory.batch import collate

    t = Transition((0,), 0, 1, (1,), False)
    assert collate([t], dtype=torch.float64).observations.shape == (1, 1)
    assert collate([t], dtype=torch.float64).rewards.dtype == torch.float64
    with pytest.raises(ValueError):
        collate([])
    with pytest.raises(ValueError):
        collate([t], dtype=torch.int64)
    with pytest.raises(ValueError):
        collate([t, Transition((0, 1), 0, 0, (1, 2), False)])
    with pytest.raises(ValueError):
        collate([Transition((1e100,), 0, 0, (0,), False)])
