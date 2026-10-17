"""Self-speculative multi-token-prediction generation trace."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from hale_llm.models.models_mtp.sample import SpeculativeStep


@dataclass
class MTPTrace:
    """Generated ids plus the draft / verify record of every forward."""
    ids: torch.Tensor
    steps: list[SpeculativeStep]
    tokens_per_forward: float
