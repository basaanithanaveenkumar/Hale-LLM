"""One pre-norm decoder block: attention, then a feed-forward network."""

from __future__ import annotations

import torch
from torch import nn

from hale_llm.nn.attention import MultiHeadAttention
from hale_llm.nn.norm import build_norm
from hale_llm.nn.positional.positional_encoding import PositionalEncoding


class TransformerBlock(nn.Module):
    """Attention and feed-forward, each with a norm before it and a residual after it.

    The feed-forward network is passed in, so the same block serves a dense MLP and a
    mixture of experts.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        ffn: nn.Module,
        dropout: float = 0.0,
        causal: bool = True,
        norm_type: str = "layer",
        positional: PositionalEncoding | None = None,
    ) -> None:
        super().__init__()
        self.attn = MultiHeadAttention(d_model, n_heads, dropout, causal, positional)
        self.norm1 = build_norm(norm_type, d_model)  # before attention
        self.ffn = ffn
        self.norm2 = build_norm(norm_type, d_model)  # before the feed-forward network

    def forward(
        self,
        inputs: torch.Tensor,
        positions: torch.Tensor | None = None,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """``[B, S, D]`` in and out."""
        # 1. Attention block, with a residual connection.
        attn_output = self.attn(self.norm1(inputs), positions, key_padding_mask)  # [B, S, D]
        x = inputs + attn_output

        # 2. Feed-forward block, with a residual connection.
        ffn_output = self.ffn(self.norm2(x))  # [B, S, D]
        return x + ffn_output
