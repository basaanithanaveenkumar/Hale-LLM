"""One extra multi-token-prediction head: predicts the token k + 1 positions ahead."""

from __future__ import annotations

import torch
import torch.nn as nn


class MTPHead(nn.Module):
    """Extra prediction head: pre-norm residual MLP on the trunk state, then a LayerNorm.

    The output goes through the backbone's (shared) LM head, so a head adds ``O(d * d_ff)``
    parameters instead of a vocabulary-sized matrix.
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm_in = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_ff, d_model)
        )
        self.norm_out = nn.LayerNorm(d_model)
        nn.init.zeros_(self.mlp[-1].weight)  # start as the identity: every head = next-token head
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """Residual pre-norm MLP over the trunk's hidden states (identity at initialisation)."""
        return self.norm_out(hidden + self.mlp(self.norm_in(hidden)))
