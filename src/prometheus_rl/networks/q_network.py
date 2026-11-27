"""Fully connected action-value approximation and fixed-target regression."""

from collections.abc import Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class QNetwork(nn.Module):
    """Map one observation [n] or a batch [B,n] to unbounded Q-values."""

    def __init__(
        self, observation_dim: int, action_dim: int,
        hidden_dims: Sequence[int] = (128, 128),
    ) -> None:
        super().__init__()
        hidden = tuple(hidden_dims)
        dims = (observation_dim, *hidden, action_dim)
        if any(type(d) is not int or d <= 0 for d in dims):
            raise ValueError("network dimensions must be positive integers")
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        layers: list[nn.Module] = []
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:], strict=True)):
            layers.append(nn.Linear(in_dim, out_dim))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
        self.model = nn.Sequential(*layers)

    def forward(self, observations: Tensor) -> Tensor:
        if observations.ndim not in (1, 2) or (
            observations.shape[-1] != self.observation_dim
        ) or (observations.ndim == 2 and observations.shape[0] == 0):
            raise ValueError("observations must have shape [n] or [B,n], B > 0")
        if not observations.is_floating_point():
            raise ValueError("observations must be floating point")
        return self.model(observations)


def select_q(values: Tensor, actions: Tensor) -> Tensor:
    """Select exactly one action value per row of a [B,A] tensor."""
    if values.ndim != 2 or min(values.shape) == 0 or not values.is_floating_point():
        raise ValueError("values must be a nonempty floating point [B,A] tensor")
    if actions.shape != (values.shape[0],) or actions.dtype != torch.long:
        raise ValueError("actions must be a long tensor of shape [B]")
    if actions.device != values.device:
        raise ValueError("actions and values must share a device")
    if torch.any((actions < 0) | (actions >= values.shape[1])):
        raise ValueError("action index out of bounds")
    return values.gather(1, actions.unsqueeze(1)).squeeze(1)


def fixed_target_loss(values: Tensor, actions: Tensor, targets: Tensor) -> Tensor:
    """Mean unit-threshold Huber loss; never backpropagate into targets."""
    predicted = select_q(values, actions)
    if targets.shape != predicted.shape or not targets.is_floating_point():
        raise ValueError("targets must be a floating point tensor of shape [B]")
    if targets.device != predicted.device or targets.dtype != predicted.dtype:
        raise ValueError("targets must match the prediction device and dtype")
    return F.smooth_l1_loss(predicted, targets.detach(), beta=1.0)
