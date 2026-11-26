"""The dense feed-forward network of a transformer block."""

from __future__ import annotations

import torch
from torch import nn

from hale_llm.nn.activation import get_activation


class FeedForward(nn.Module):
    """``Linear -> activation -> Linear -> Dropout``: widen to ``d_ff``, mix, narrow back.

    ``activation`` is a name from :mod:`hale_llm.nn.activation`. A gated one (``swiglu``, ``geglu``, ``reglu``)
    multiplies the activated vector by a second projection of the input, which adds one ``d_model x d_ff`` matrix.
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0, activation: str = "gelu") -> None:
        super().__init__()
        spec = get_activation(activation)
        self.activation = spec.fn
        self.up = nn.Linear(d_model, d_ff)
        self.down = nn.Linear(d_ff, d_model)
        # Created after ``down`` so that the plain (gelu) layer keeps the parameter order of old checkpoints.
        self.gate = nn.Linear(d_model, d_ff) if spec.gated else None
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``[B, S, D]`` in and out."""
        hidden = self.activation(self.up(x))  # [B, S, d_ff]
        if self.gate is not None:
            hidden = hidden * self.gate(x)  # [B, S, d_ff] the gate decides how much of each unit passes
        return self.dropout(self.down(hidden))  # [B, S, D]
