"""Multi-head attention: several :class:`HeadAttention` heads, concatenated and projected."""

from __future__ import annotations

import torch
from torch import nn

from hale_llm.nn.attention.head_attention import HeadAttention
from hale_llm.nn.positional.positional_encoding import PositionalEncoding


class MultiHeadAttention(nn.Module):
    """``n_heads`` independent heads of width ``d_model // n_heads``."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.0,
        causal: bool = False,
        positional: PositionalEncoding | None = None,
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model={d_model} must be divisible by n_heads={n_heads}")
        head_size = d_model // n_heads
        self.heads = nn.ModuleList(
            HeadAttention(d_model, head_size, dropout=dropout, causal=causal, positional=positional)
            for _ in range(n_heads)
        )
        self.proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        inputs: torch.Tensor,
        positions: torch.Tensor | None = None,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """``[B, S, D]`` in and out."""
        # 1. Every head attends on its own.
        head_outputs = [head(inputs, positions, key_padding_mask) for head in self.heads]  # n x [B, S, H]

        # 2. Concatenate the heads back to the model width.
        concat_output = torch.cat(head_outputs, dim=-1)  # [B, S, D]

        # 3. Mix them with one output projection.
        output = self.proj(concat_output)  # [B, S, D]
        return self.dropout(output)
