"""The dense feed-forward network of a transformer block."""

from __future__ import annotations

import torch
from torch import nn


class FeedForward(nn.Module):
    """``Linear -> GELU -> Linear -> Dropout``."""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
