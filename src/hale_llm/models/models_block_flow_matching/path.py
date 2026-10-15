"""Discrete masking probability path for block flow matching.

A token is drawn from the data with probability ``kappa(t)`` and is the mask token otherwise:
``p_t(x) = kappa(t) * delta(x_1) + (1 - kappa(t)) * delta(mask)``, ``kappa(0) = 0``,
``kappa(1) = 1``. The mixture path's generating rate moves a masked token to the predicted data
token at rate ``kappa'(t) / (1 - kappa(t))``; that ratio is both the Euler unmasking
probability per step and the weight of the cross-entropy at time ``t`` (Gat et al., 2024).

Each block draws its own ``t``, so one sequence holds blocks at different points of the flow -
the conditioning the sampler meets when it integrates block ``b`` given clean blocks ``< b``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import torch

Schedule = Literal["linear", "cosine"]


@dataclass(frozen=True)
class MaskingPath:
    """``kappa`` schedule of the masking path.

    Attributes:
        schedule: ``linear`` (``kappa = t``) or ``cosine`` (``kappa = 1 - cos(pi t / 2)``: few
            tokens revealed early, most near ``t = 1``).
        eps: keeps ``t`` away from 1, where the rate ``kappa' / (1 - kappa)`` diverges.
    """

    schedule: Schedule = "linear"
    eps: float = 1e-3

    def __post_init__(self) -> None:
        if self.schedule not in ("linear", "cosine"):
            raise ValueError(f"unknown masking schedule {self.schedule!r}")

    def kappa(self, t: torch.Tensor | float) -> torch.Tensor | float:
        """Probability that a token is already data at time ``t``."""
        if self.schedule == "linear":
            return t
        if torch.is_tensor(t):
            return 1.0 - torch.cos(0.5 * math.pi * t)
        return 1.0 - math.cos(0.5 * math.pi * t)

    def rate(self, t: torch.Tensor) -> torch.Tensor:
        """``kappa'(t) / (1 - kappa(t))``: the unmasking rate and the loss weight at ``t``."""
        t = t.clamp(max=1.0 - self.eps)
        if self.schedule == "linear":
            return 1.0 / (1.0 - t)
        return 0.5 * math.pi * torch.tan(0.5 * math.pi * t)

    def unmask_probability(self, t_cur: float, t_next: float) -> float:
        """Exact probability that a token masked at ``t_cur`` is data at ``t_next``."""
        k_cur, k_next = float(self.kappa(t_cur)), float(self.kappa(t_next))
        if k_cur >= 1.0 - 1e-8:
            return 1.0
        return min(max((k_next - k_cur) / (1.0 - k_cur), 0.0), 1.0)

    def sample_times(self, batch: int, num_blocks: int, device: torch.device) -> torch.Tensor:
        """``(batch, num_blocks)`` times ``t ~ U[0, 1 - eps)``, one per block."""
        return torch.rand(batch, num_blocks, device=device) * (1.0 - self.eps)


def corrupt_blocks(
    x1: torch.Tensor, block_size: int, mask_token_id: int, path: MaskingPath
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample ``x_t`` from the path with an independent ``t`` per block.

    Returns:
        x_t: ``(batch, seq)`` noised tokens (mask token where not yet revealed).
        is_masked: ``(batch, seq)`` positions that are masked (the loss positions).
        t: ``(batch, seq)`` the time of each position's block.
    """
    b, n = x1.shape
    if n % block_size != 0:
        raise ValueError(f"seq_len {n} must be divisible by block_size {block_size}")
    t = path.sample_times(b, n // block_size, x1.device).repeat_interleave(block_size, dim=1)
    is_masked = torch.rand(b, n, device=x1.device) >= path.kappa(t)
    x_t = torch.where(is_masked, torch.full_like(x1, mask_token_id), x1)
    return x_t, is_masked, t
