"""The output of one :meth:`InferenceSession.generate` call."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class GenerationResult:
    token_ids: torch.Tensor
    text: str
    prompt: str
    history: list | None = None
