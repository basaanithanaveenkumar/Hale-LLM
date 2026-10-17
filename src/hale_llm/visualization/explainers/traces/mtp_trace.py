"""Self-speculative multi-token-prediction generation trace."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from hale_llm.models.models_mtp.sample import SpeculativeStep


@dataclass
class MTPTrace:
    ids: torch.Tensor
    steps: list[SpeculativeStep]
    tokens_per_forward: float
