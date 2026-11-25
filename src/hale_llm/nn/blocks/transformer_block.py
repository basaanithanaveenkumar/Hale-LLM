"""One pre-norm decoder block: attention, then a feed-forward network."""

from __future__ import annotations

import torch
from torch import nn

from hale_llm.nn.attention import MultiHeadAttention
from hale_llm.nn.norm import build_norm
from hale_llm.nn.positional.positional_encoding import PositionalEncoding


class TransformerBlock(nn.Module):
    """``x + attention(norm(x))`` followed by ``x + ffn(norm(x))``.

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
        self.norm1 = build_norm(norm_type, d_model)
        self.ffn = ffn
        self.norm2 = build_norm(norm_type, d_model)

    def forward(
        self,
        inputs: torch.Tensor,
        positions: torch.Tensor | None = None,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = inputs + self.attn(self.norm1(inputs), positions, key_padding_mask)
        return x + self.ffn(self.norm2(x))

    def auxiliary_loss(self) -> torch.Tensor:
        """The feed-forward network's extra training loss (MoE load balancing), else 0."""
        aux = getattr(self.ffn, "aux_loss", None)
        return aux if aux is not None else torch.zeros((), device=next(self.parameters()).device)
