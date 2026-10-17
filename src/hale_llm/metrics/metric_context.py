"""What a metric sees for one evaluation batch."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn


@dataclass
class MetricContext:
    """Everything a metric may read for one evaluation batch.

    Attributes:
        model: the model under evaluation (eval mode, no grad).
        batch: the collated batch already on the model's device.
        loss: the variant's loss on this batch.
        variant: registered variant name (metrics may branch on it).
    """
    model: nn.Module
    batch: dict[str, Any]
    loss: torch.Tensor
    variant: str
