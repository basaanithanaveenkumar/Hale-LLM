"""The dense feed-forward network of a transformer block."""

from __future__ import annotations

import torch
from torch import nn


class FeedForward(nn.Module):
    """``Linear -> GELU -> Linear -> Dropout``: widen to ``d_ff``, mix, narrow back."""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.up = nn.Linear(d_model, d_ff)
        self.act = nn.GELU()
        self.down = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``[B, S, D]`` in and out."""
        hidden = self.act(self.up(x))  # [B, S, d_ff]
        return self.dropout(self.down(hidden))  # [B, S, D]
