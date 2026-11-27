"""Convert immutable replay samples to the neural network tensor contract."""

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor

from prometheus_rl.memory.replay import Transition


@dataclass(frozen=True, slots=True)
class TransitionBatch:
    observations: Tensor
    actions: Tensor
    rewards: Tensor
    next_observations: Tensor
    terminated: Tensor
    truncated: Tensor

    @property
    def bootstrap_mask(self) -> Tensor:
        return (~self.terminated).to(self.rewards.dtype)

    @property
    def episode_end(self) -> Tensor:
        return self.terminated | self.truncated


def collate(
    transitions: Sequence[Transition], *,
    device: str | torch.device = "cpu", dtype: torch.dtype = torch.float32,
) -> TransitionBatch:
    """Create [B,n] observations and [B] scalar fields, without gradients."""
    if not transitions:
        raise ValueError("cannot collate an empty batch")
    if dtype not in (torch.float32, torch.float64):
        raise ValueError("dtype must be float32 or float64")
    dim = len(transitions[0].observation)
    if any(len(t.observation) != dim for t in transitions):
        raise ValueError("observation dimensions must match")
    observations = torch.tensor([t.observation for t in transitions],
                                dtype=dtype, device=device)
    next_observations = torch.tensor([t.next_observation for t in transitions],
                                     dtype=dtype, device=device)
    rewards = torch.tensor([t.reward for t in transitions],
                           dtype=dtype, device=device)
    if not all(torch.isfinite(x).all().item()
               for x in (observations, next_observations, rewards)):
        raise ValueError("transition values overflow the requested dtype")
    return TransitionBatch(
        observations,
        torch.tensor([t.action for t in transitions], dtype=torch.long, device=device),
        rewards,
        next_observations,
        torch.tensor([t.terminated for t in transitions], dtype=torch.bool,
                     device=device),
        torch.tensor([t.truncated for t in transitions], dtype=torch.bool,
                     device=device),
    )
