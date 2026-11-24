"""Common interface for every positional-encoding scheme."""

from __future__ import annotations

from abc import ABC

import torch
from torch import nn


class PositionalEncoding(nn.Module, ABC):
    """Tells the model where each token sits in the sequence.

    Schemes differ in *where* they act, so the interface has two hooks and each
    scheme overrides the one it needs; the other one passes data through:

    * :meth:`encode_embeddings` runs once on the token embeddings (learned or
      sinusoidal tables add a vector here).
    * :meth:`encode_queries_keys` runs inside every attention layer (rotary
      embeddings rotate the query and key vectors here).
    """

    #: Short name used in config files and in log lines.
    name: str = "base"

    def encode_embeddings(self, hidden: torch.Tensor) -> torch.Tensor:
        """Hook for schemes that add position to the token embeddings. Default: no-op."""
        return hidden

    def encode_queries_keys(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Hook for schemes that act on attention queries and keys. Default: no-op.

        Args:
            q: queries, shape ``(batch, heads, seq, head_dim)``.
            k: keys, shape ``(batch, kv_heads, seq, head_dim)``.
            positions: integer position of each token, shape ``(seq,)``.
        """
        return q, k

    def describe(self) -> str:
        """One-line summary written to the log when a model is built."""
        return self.name
