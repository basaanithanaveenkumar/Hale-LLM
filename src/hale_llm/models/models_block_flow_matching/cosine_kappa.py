"""Cosine ``kappa`` schedule."""

from __future__ import annotations

import math

import torch

from hale_llm.models.models_block_flow_matching.kappa_schedule import (
    KappaSchedule,
    register_kappa_schedule,
)


@register_kappa_schedule("cosine")
class CosineKappa(KappaSchedule):
    """Strategy ``kappa(t) = 1 - cos(pi t / 2)``: few tokens revealed early, most near ``t = 1``."""

    def kappa(self, t: torch.Tensor | float) -> torch.Tensor | float:
        """Return ``1 - cos(pi t / 2)``."""
        if torch.is_tensor(t):
            return 1.0 - torch.cos(0.5 * math.pi * t)
        return 1.0 - math.cos(0.5 * math.pi * t)

    def rate(self, t: torch.Tensor) -> torch.Tensor:
        """Return ``(pi / 2) tan(pi t / 2)``."""
        return 0.5 * math.pi * torch.tan(0.5 * math.pi * t)
