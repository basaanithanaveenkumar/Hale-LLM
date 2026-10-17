"""Full looped-model generation trace."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from hale_llm.visualization.explainers.traces.loop_gen_step import LoopGenStep
from hale_llm.visualization.explainers.traces.loop_pass import LoopPass


@dataclass
class LoopedTrace:
    """Generated ids with every loop pass and every generation step."""
    ids: torch.Tensor
    passes: list[LoopPass] = field(default_factory=list)
    steps: list[LoopGenStep] = field(default_factory=list)
