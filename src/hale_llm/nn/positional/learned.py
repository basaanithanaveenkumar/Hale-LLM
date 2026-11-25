"""Learned absolute position embeddings (GPT-2 style)."""

from __future__ import annotations

import torch
from torch import nn

from hale_llm.nn.positional.positional_encoding import PositionalEncoding
from hale_llm.nn.positional.registry import register_positional_encoding


@register_positional_encoding("learned")
class LearnedPositionalEncoding(PositionalEncoding):
    """One trainable vector per position, added to the token embeddings.

    Costs ``max_length * d_model`` parameters and cannot be used beyond
    ``max_length`` positions.
    """

    name = "learned"

    def __init__(self, d_model: int, max_length: int, **_unused) -> None:
        super().__init__()
        self.max_length = max_length
        self.table = nn.Parameter(torch.randn(1, max_length, d_model) * 0.02)

    def encode_embeddings(self, hidden: torch.Tensor) -> torch.Tensor:
        seq_len = hidden.size(1)
        if seq_len > self.max_length:
            raise ValueError(f"sequence of {seq_len} tokens exceeds max_length={self.max_length}")
        return hidden + self.table[:, :seq_len]

    def describe(self) -> str:
        return f"learned(max_length={self.max_length}, params={self.table.numel():,})"
