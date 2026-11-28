"""Bounded byte-valued replay; copies both stacks to avoid aliasing."""

from random import Random
from typing import NamedTuple

import numpy as np
import torch
from torch import Tensor

from prometheus_rl.memory.batch import TransitionBatch
from prometheus_rl.networks.visual import PIXEL_SHAPE


class PixelExperience(NamedTuple):
    observation: np.ndarray
    action: int
    reward: float
    next_observation: np.ndarray
    terminated: bool
    truncated: bool


def validate_pixels(pixels: np.ndarray) -> np.ndarray:
    array = np.asarray(pixels)
    if array.shape != PIXEL_SHAPE or array.dtype != np.uint8:
        raise ValueError("expected uint8 observation with shape (4, 84, 84)")
    return array


class PixelReplay:
    def __init__(self, capacity: int, actions: int, *, seed: int = 0) -> None:
        if any(type(x) is not int or x < 1 for x in (capacity, actions)):
            raise ValueError("capacity and actions must be positive integers")
        self.capacity = capacity
        self.actions = actions
        self._rng = Random(seed)
        self._items: list[PixelExperience] = []
        self._cursor = 0

    def __len__(self) -> int:
        return len(self._items)

    @property
    def pixel_bytes(self) -> int:
        return len(self) * 2 * int(np.prod(PIXEL_SHAPE))

    def append(self, item: PixelExperience) -> None:
        if type(item.action) is not int or not 0 <= item.action < self.actions:
            raise ValueError("invalid action")
        if not np.isfinite(item.reward) or abs(item.reward) > np.finfo(np.float32).max:
            raise ValueError("nonfinite reward")
        if type(item.terminated) is not bool or type(item.truncated) is not bool:
            raise ValueError("flags must be bool")
        first = validate_pixels(item.observation).copy()
        second = validate_pixels(item.next_observation).copy()
        first.setflags(write=False)
        second.setflags(write=False)
        entry = PixelExperience(first, item.action, float(item.reward), second,
                                item.terminated, item.truncated)
        if len(self) < self.capacity:
            self._items.append(entry)
        else:
            self._items[self._cursor] = entry
        self._cursor = (self._cursor + 1) % self.capacity

    def sample(
        self, size: int, *, device: str | torch.device = "cpu"
    ) -> TransitionBatch:
        if type(size) is not int or not 1 <= size <= len(self):
            raise ValueError("invalid sample size")
        items = self._rng.sample(self._items, size)

        def tensor(values, dtype: torch.dtype) -> Tensor:
            return torch.as_tensor(values, dtype=dtype, device=device)

        return TransitionBatch(
            observations=tensor(np.stack([i.observation for i in items]), torch.uint8),
            actions=tensor([i.action for i in items], torch.long),
            rewards=tensor([i.reward for i in items], torch.float32),
            next_observations=tensor(
                np.stack([i.next_observation for i in items]), torch.uint8
            ),
            terminated=tensor([i.terminated for i in items], torch.bool),
            truncated=tensor([i.truncated for i in items], torch.bool),
        )
