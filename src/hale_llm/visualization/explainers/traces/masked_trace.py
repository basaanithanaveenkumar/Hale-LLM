"""Full masked-generation trace (one :class:`MaskedStep` per denoising step)."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from hale_llm.visualization.explainers.traces.masked_step import MaskedStep


@dataclass
class MaskedTrace:
    ids: torch.Tensor
    steps: list[MaskedStep] = field(default_factory=list)
