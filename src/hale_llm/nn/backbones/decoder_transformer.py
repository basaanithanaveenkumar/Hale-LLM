"""A stack of transformer blocks followed by a final norm."""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import nn

from hale_llm.nn.blocks import TransformerBlock
from hale_llm.nn.norm import build_norm
from hale_llm.nn.positional.positional_encoding import PositionalEncoding


class DecoderTransformer(nn.Module):
    """Hidden states in, hidden states out (no embeddings, no output head)."""

    def __init__(
        self,
        n_layers: int,
        d_model: int,
        n_heads: int,
        ffn_factory: Callable[[], nn.Module],
        dropout: float = 0.0,
        norm_type: str = "layer",
        positional: PositionalEncoding | None = None,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            TransformerBlock(
                d_model,
                n_heads,
                ffn=ffn_factory(),
                dropout=dropout,
                causal=True,
                norm_type=norm_type,
                positional=positional,
            )
            for _ in range(n_layers)
        )
        self.norm = build_norm(norm_type, d_model)

    def forward(
        self,
        inputs: torch.Tensor,
        positions: torch.Tensor | None = None,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = inputs
        for layer in self.layers:
            x = layer(x, positions, key_padding_mask)
        return self.norm(x)

    def auxiliary_loss(self) -> torch.Tensor:
        """Sum of every block's extra loss (zero for a dense network)."""
        return torch.stack([layer.auxiliary_loss() for layer in self.layers]).sum()
