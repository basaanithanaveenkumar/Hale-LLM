"""The output of one :meth:`InferenceSession.generate` call."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class GenerationResult:
    """One generated completion.

    Attributes:
        token_ids: ``(1, L)`` prompt plus generated ids.
        text: decoded text (special tokens skipped).
        prompt: the prompt used.
        history: per-step snapshots from the sampler when requested, else ``None``.
    """

    token_ids: torch.Tensor
    text: str
    prompt: str
    history: list | None = None
