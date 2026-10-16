"""Loss and diagnostics returned by :class:`GRPOObjective`."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class GRPOLossTerms:
    """The loss and its diagnostics for one inner step."""

    loss: torch.Tensor
    policy_loss: float
    kl: float
    clip_fraction: float
    ratio_mean: float
