"""One feed-forward expert."""

from __future__ import annotations

import torch
from torch import nn

from hale_llm.nn.activation import get_activation


class Expert(nn.Module):
    """A small feed-forward network.

    * plain activation (``gelu``, ``relu``, ``silu``): ``w2( act(w1 x) )``.
    * gated activation (``swiglu``, ``geglu``, ``reglu``): ``w2( act(w1 x) * w3 x )``, one matrix more.
    """

    def __init__(
        self, d_model: int, hidden_dim: int, dropout: float = 0.0, activation: str = "swiglu"
    ) -> None:
        super().__init__()
        spec = get_activation(activation)  # KeyError lists the registered names
        self.activation = spec.fn
        self.w1 = nn.Linear(d_model, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, d_model, bias=False)
        self.w3 = nn.Linear(d_model, hidden_dim, bias=False) if spec.gated else None
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``[T, D]`` in and out."""
        hidden = self.activation(self.w1(x))  # [T, hidden_dim]
        if self.w3 is not None:
            hidden = hidden * self.w3(x)  # [T, hidden_dim] the gate decides how much of each unit passes
        return self.dropout(self.w2(hidden))  # [T, D]
