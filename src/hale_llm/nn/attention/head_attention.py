"""One attention head: scaled dot-product attention over a ``head_size`` slice."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from hale_llm.nn.positional.positional_encoding import PositionalEncoding


class HeadAttention(nn.Module):
    """Projects ``(batch, seq, d_model)`` to one query/key/value head and attends.

    Steps in :meth:`forward`:

    1. project the input to query, key and value of width ``head_size`` (no bias),
    2. let the positional encoding act on queries and keys (RoPE rotates them; a learned
       table acts on the embeddings instead, so this step does nothing for it),
    3. score every query against every key, scaled by ``1 / sqrt(head_size)``,
    4. hide the future (``causal``) and padded keys, softmax, and weight the values.

    The scale uses the width of the vectors being multiplied (``head_size``), not the
    model width.

    Two ways to run steps 3 and 4:

    * default: PyTorch's fused kernel, which never builds the ``(seq, seq)`` matrix;
    * ``return_attn_weights=True``: an explicit softmax that also returns the weights,
      for inspection. Slower and uses more memory.
    """

    def __init__(
        self,
        d_model: int,
        head_size: int,
        dropout: float = 0.0,
        causal: bool = False,
        return_attn_weights: bool = False,
        positional: PositionalEncoding | None = None,
    ) -> None:
        super().__init__()
        self.causal = causal
        self.dropout_p = dropout
        self.return_attn_weights = return_attn_weights
        self.positional = positional  # shared with the backbone; RoPE has no parameters
        self.scale = 1.0 / math.sqrt(head_size)
        self.query_proj = nn.Linear(d_model, head_size, bias=False)
        self.key_proj = nn.Linear(d_model, head_size, bias=False)
        self.value_proj = nn.Linear(d_model, head_size, bias=False)

    def forward(
        self,
        inputs: torch.Tensor,
        positions: torch.Tensor | None = None,
        key_padding_mask: torch.Tensor | None = None,
    ):
        """Return ``(batch, seq, head_size)``, or ``(output, weights)`` if weights are requested.

        ``key_padding_mask`` is ``(batch, seq)``, ``True`` where a key must be ignored.
        """
        query = self.query_proj(inputs)
        key = self.key_proj(inputs)
        value = self.value_proj(inputs)
        query, key = self._add_positions(query, key, positions)

        if self.return_attn_weights:
            return self._attend_explicitly(query, key, value, key_padding_mask)
        return self._attend_fused(query, key, value, key_padding_mask)

    def _add_positions(self, query, key, positions):
        if self.positional is None:
            return query, key
        if positions is None:
            positions = torch.arange(query.size(1), device=query.device)
        # The positional hook expects a heads axis: (batch, heads=1, seq, head_size).
        query, key = self.positional.encode_queries_keys(
            query.unsqueeze(1), key.unsqueeze(1), positions
        )
        return query.squeeze(1), key.squeeze(1)

    def _blocked(self, seq_len: int, key_padding_mask, device) -> torch.Tensor:
        """Boolean ``(batch or 1, seq, seq)`` mask, ``True`` where attention is not allowed."""
        blocked = torch.zeros(1, seq_len, seq_len, dtype=torch.bool, device=device)
        if self.causal:  # a query may not look at keys after it
            blocked = blocked | torch.ones_like(blocked[0]).triu(diagonal=1)
        if key_padding_mask is not None:
            blocked = blocked | key_padding_mask[:, None, :]
        return blocked

    def _attend_fused(self, query, key, value, key_padding_mask):
        if key_padding_mask is None:
            # Pure causal (or no) masking: let the kernel build the mask itself.
            return F.scaled_dot_product_attention(
                query, key, value,
                dropout_p=self.dropout_p if self.training else 0.0,
                is_causal=self.causal,
                scale=self.scale,
            )
        blocked = self._blocked(query.size(1), key_padding_mask, query.device)
        # A large finite negative (not -inf) keeps all-padding rows from turning into NaN.
        bias = torch.zeros(blocked.shape, dtype=query.dtype, device=query.device)
        bias = bias.masked_fill(blocked, torch.finfo(query.dtype).min)
        return F.scaled_dot_product_attention(
            query, key, value,
            attn_mask=bias,
            dropout_p=self.dropout_p if self.training else 0.0,
            scale=self.scale,
        )

    def _attend_explicitly(self, query, key, value, key_padding_mask):
        scores = torch.matmul(query, key.transpose(-2, -1)) * self.scale
        blocked = self._blocked(query.size(1), key_padding_mask, query.device)
        weights = F.softmax(scores.masked_fill(blocked, float("-inf")), dim=-1)
        weights = F.dropout(weights, p=self.dropout_p, training=self.training)
        return torch.matmul(weights, value), weights
