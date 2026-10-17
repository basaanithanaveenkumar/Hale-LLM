"""What a metric sees for one evaluation batch."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn


@dataclass
class MetricContext:
    model: nn.Module
    batch: dict[str, Any]
    loss: torch.Tensor
    variant: str
