"""One feed-forward expert."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class Expert(nn.Module):
    """``w2(act(w1 x) * w3 x)`` for SwiGLU, or ``w2(gelu(w1 x))`` for plain GELU.

    SwiGLU uses a third matrix as a gate, so at the same ``hidden_dim`` it has 50%
    more parameters than the GELU form.
    """

    def __init__(
        self, d_model: int, hidden_dim: int, dropout: float = 0.0, activation: str = "swiglu"
    ) -> None:
        super().__init__()
        if activation not in ("swiglu", "gelu"):
            raise ValueError(f"unknown expert activation {activation!r}; use swiglu or gelu")
        self.activation = activation
        self.w1 = nn.Linear(d_model, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, d_model, bias=False)
        self.w3 = nn.Linear(d_model, hidden_dim, bias=False) if activation == "swiglu" else None
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.w3 is not None:
            hidden = F.silu(self.w1(x)) * self.w3(x)
        else:
            hidden = F.gelu(self.w1(x))
        return self.dropout(self.w2(hidden))
