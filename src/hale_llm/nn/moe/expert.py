"""One feed-forward expert."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class Expert(nn.Module):
    """A small feed-forward network.

    * ``swiglu``: ``w2( silu(w1 x) * w3 x )``, where ``w3 x`` gates the hidden vector.
    * ``gelu``:   ``w2( gelu(w1 x) )``, one matrix fewer.
    """

    def __init__(
        self, d_model: int, hidden_dim: int, dropout: float = 0.0, activation: str = "swiglu"
    ) -> None:
        super().__init__()
        if activation not in ("swiglu", "gelu"):
            raise ValueError(f"unknown expert activation {activation!r}; use swiglu or gelu")
        self.w1 = nn.Linear(d_model, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, d_model, bias=False)
        self.w3 = nn.Linear(d_model, hidden_dim, bias=False) if activation == "swiglu" else None
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``[T, D]`` in and out."""
        if self.w3 is not None:
            hidden = F.silu(self.w1(x)) * self.w3(x)  # [T, hidden_dim]
        else:
            hidden = F.gelu(self.w1(x))  # [T, hidden_dim]
        return self.dropout(self.w2(hidden))  # [T, D]
