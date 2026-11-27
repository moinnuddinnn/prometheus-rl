"""Neural tests run with the optional PyTorch dependency."""

import pytest

# The dependency-light test job intentionally skips this module.
torch = pytest.importorskip("torch")

from prometheus_rl.networks import QNetwork, fixed_target_loss, select_q  # noqa: E402
from prometheus_rl.networks.demo import run_demo  # noqa: E402


def test_cartpole_shapes_and_parameter_count():
    model = QNetwork(4, 2)
    assert model(torch.zeros(4)).shape == (2,)
    assert model(torch.zeros(7, 4)).shape == (7, 2)
    assert sum(p.numel() for p in model.parameters()) == 17410


def test_linear_output_can_be_negative():
    model = QNetwork(2, 2, ())
    with torch.no_grad():
        model.model[0].weight.zero_()
        model.model[0].bias.copy_(torch.tensor([-3., 2.]))
    assert model(torch.zeros(2)).tolist() == [-3., 2.]


@pytest.mark.parametrize("shape", [(3,), (0, 4), (2, 3), (2, 2, 4), ()])
def test_reject_bad_observation_shapes(shape):
    with pytest.raises(ValueError):
        QNetwork(4, 2)(torch.zeros(shape))


@pytest.mark.parametrize("dims", [(0, 2, ()), (4, -1, ()), (4, 2, (0,))])
def test_reject_bad_dimensions(dims):
    with pytest.raises(ValueError):
        QNetwork(*dims)


def test_reject_integer_observations():
    with pytest.raises(ValueError):
        QNetwork(4, 2)(torch.zeros(4, dtype=torch.long))


def test_gather_and_gradient_routing():
    values = torch.tensor([[1., 2.], [3., 4.]], requires_grad=True)
    selected = select_q(values, torch.tensor([1, 0]))
    assert selected.tolist() == [2., 3.]
    selected.sum().backward()
    assert values.grad.tolist() == [[0., 1.], [1., 0.]]


@pytest.mark.parametrize("actions", [[-1, 0], [2, 0], [0], [[0], [1]], [0., 1.]])
def test_reject_invalid_actions(actions):
    with pytest.raises(ValueError):
        select_q(torch.zeros(2, 2), torch.tensor(actions))


def test_huber_value_gradient_and_detached_targets():
    values = torch.tensor([[0.5, 9.], [9., 2.]], requires_grad=True)
    targets = torch.zeros(2, requires_grad=True)
    loss = fixed_target_loss(values, torch.tensor([0, 1]), targets)
    assert loss.item() == pytest.approx((0.125 + 1.5) / 2)
    loss.backward()
    assert targets.grad is None
    assert values.grad.tolist() == [[0.25, 0.], [0., 0.5]]


def test_reject_target_broadcasting():
    with pytest.raises(ValueError):
        fixed_target_loss(torch.zeros(2, 2), torch.tensor([0, 1]), torch.zeros(2, 1))


def test_multilayer_gradient_matches_finite_difference():
    model = QNetwork(2, 2, (3, 2)).double()
    with torch.no_grad():
        for p in model.parameters():
            p.fill_(0.2)
    x = torch.tensor([[1., 2.]], dtype=torch.double)
    a = torch.tensor([0])
    y = torch.tensor([0.1], dtype=torch.double)
    loss = fixed_target_loss(model(x), a, y)
    loss.backward()
    p = model.model[0].weight
    analytic = p.grad[0, 0].item()
    h = 1e-6
    with torch.no_grad():
        original = p[0, 0].item()
        p[0, 0] = original + h
        plus = fixed_target_loss(model(x), a, y).item()
        p[0, 0] = original - h
        minus = fixed_target_loss(model(x), a, y).item()
        p[0, 0] = original
    assert analytic != 0
    assert analytic == pytest.approx((plus - minus) / (2 * h), rel=1e-6)


def test_fixed_target_training_reduces_loss_and_preserves_rng():
    before = torch.random.get_rng_state().clone()
    count, initial, final = run_demo()
    assert count == 17410
    assert final < initial * 0.01
    assert torch.equal(before, torch.random.get_rng_state())
    assert run_demo() == (count, initial, final)
