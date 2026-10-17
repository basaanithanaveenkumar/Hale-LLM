"""Full autoregressive generation trace (one :class:`ARStep` per token)."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from hale_llm.visualization.explainers.traces.ar_step import ARStep


@dataclass
class ARTrace:
    """Generated ids plus one :class:`ARStep` per generated token."""
    ids: torch.Tensor
    steps: list[ARStep] = field(default_factory=list)
