"""A deterministic fixed-target regression demonstration, not RL training."""

import torch

from prometheus_rl.networks import QNetwork, fixed_target_loss


def run_demo() -> tuple[int, float, float]:
    """Fit two fixed action targets on a small CPU batch using SGD."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(7)
        network = QNetwork(4, 2)
        optimizer = torch.optim.SGD(network.parameters(), lr=0.03)
        observations = torch.tensor([[1., 0., 0., 0.], [0., 1., 0., 0.]])
        actions = torch.tensor([0, 1])
        targets = torch.tensor([1., -1.])
        with torch.no_grad():
            initial = fixed_target_loss(network(observations), actions, targets).item()
        for _ in range(80):
            optimizer.zero_grad(set_to_none=True)
            loss = fixed_target_loss(network(observations), actions, targets)
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            final = fixed_target_loss(network(observations), actions, targets).item()
        return sum(p.numel() for p in network.parameters()), initial, final
