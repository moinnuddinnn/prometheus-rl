"""Convolutional action values for stacked Atari observations."""

import torch
from torch import Tensor, nn

PIXEL_SHAPE = (4, 84, 84)


class VisualQNetwork(nn.Module):
    """Nature-DQN convolutional trunk with optional dueling heads."""

    def __init__(self, actions: int, *, dueling: bool = False) -> None:
        super().__init__()
        if type(actions) is not int or actions < 1:
            raise ValueError("actions must be a positive integer")
        self.features = nn.Sequential(
            nn.Conv2d(4, 32, 8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, 3), nn.ReLU(),
            nn.Flatten(), nn.Linear(3136, 512), nn.ReLU(),
        )
        self.advantage = nn.Linear(512, actions)
        self.value = nn.Linear(512, 1) if dueling else None

    def forward(self, pixels: Tensor) -> Tensor:
        if pixels.ndim != 4 or tuple(pixels.shape[1:]) != PIXEL_SHAPE:
            raise ValueError("expected [batch, 4, 84, 84]")
        if pixels.dtype != torch.uint8:
            raise ValueError("expected uint8 pixel observations")
        h = self.features(pixels.float() / 255.0)
        a = self.advantage(h)
        if self.value is None:
            return a
        return self.value(h) + (a - a.mean(dim=1, keepdim=True))
