"""One attention head: scaled dot-product attention over a ``head_size`` slice."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from hale_llm.nn.positional.positional_encoding import PositionalEncoding


class HeadAttention(nn.Module):
    """Projects ``(batch, seq, d_model)`` into one query/key/value head and attends.

    Steps in :meth:`forward`:

    1. project the input to query, key and value of width ``head_size`` (no bias),
    2. let the positional encoding rotate or shift queries and keys (a no-op for
       schemes that act on the embeddings instead),
    3. attend with a causal mask (and an optional padding mask),
    4. weight the values.

    The scores are scaled by ``1 / sqrt(head_size)``: that is the width of the vectors
    being multiplied, not the model width.

    By default the fused PyTorch kernel does the attention, which never builds the
    ``(seq, seq)`` weight matrix. Set ``return_attn_weights`` to get the weights back
    for inspection; that switches to an explicit softmax and is slower.
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
        self.head_size = head_size
        self.causal = causal
        self.dropout_p = dropout
        self.return_attn_weights = return_attn_weights
        # Held in a tuple so the shared module is not registered under every head (it is owned,
        # moved to devices and saved by the backbone).
        self._positional = (positional,)
        self.query_proj = nn.Linear(d_model, head_size, bias=False)
        self.key_proj = nn.Linear(d_model, head_size, bias=False)
        self.value_proj = nn.Linear(d_model, head_size, bias=False)
        self.scale = 1.0 / math.sqrt(head_size)

    def forward(
        self,
        inputs: torch.Tensor,
        positions: torch.Tensor | None = None,
        key_padding_mask: torch.Tensor | None = None,
    ):
        """Return ``(batch, seq, head_size)``, or ``(output, weights)`` if weights are requested.

        ``key_padding_mask`` is ``(batch, seq)`` with ``True`` where a key must be ignored.
        """
        batch, seq_len, _ = inputs.shape
        query = self.query_proj(inputs)
        key = self.key_proj(inputs)
        value = self.value_proj(inputs)

        positional = self._positional[0]
        if positional is not None:
            if positions is None:
                positions = torch.arange(seq_len, device=inputs.device)
            # The positional hooks expect a heads axis: (batch, heads=1, seq, head_size).
            query, key = positional.encode_queries_keys(
                query.unsqueeze(1), key.unsqueeze(1), positions
            )
            query, key = query.squeeze(1), key.squeeze(1)

        blocked = self._blocked_mask(seq_len, key_padding_mask, inputs.device)
        if self.return_attn_weights:
            return self._explicit_attention(query, key, value, blocked)

        additive = None
        if blocked is not None:
            additive = torch.zeros(blocked.shape, dtype=query.dtype, device=query.device)
            additive = additive.masked_fill(blocked, torch.finfo(query.dtype).min)
        return F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=additive,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=self.causal and additive is None,
            scale=self.scale,
        )

    def _blocked_mask(
        self, seq_len: int, key_padding_mask: torch.Tensor | None, device: torch.device
    ) -> torch.Tensor | None:
        """Boolean ``True = may not attend`` mask, or ``None`` when nothing is blocked.

        Causal-only attention returns ``None`` so the fused kernel can use ``is_causal``.
        """
        if key_padding_mask is None:
            return None if not self.return_attn_weights else self._causal_blocked(seq_len, device)
        blocked = key_padding_mask[:, None, :]
        if self.causal:
            blocked = blocked | self._causal_blocked(seq_len, device)[None]
        return blocked

    def _causal_blocked(self, seq_len: int, device: torch.device) -> torch.Tensor:
        if not self.causal:
            return torch.zeros(seq_len, seq_len, dtype=torch.bool, device=device)
        return torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device), diagonal=1)

    def _explicit_attention(self, query, key, value, blocked):
        scores = torch.matmul(query, key.transpose(-2, -1)) * self.scale
        if blocked is not None:
            scores = scores.masked_fill(blocked, float("-inf"))
        weights = F.softmax(scores, dim=-1)
        weights = F.dropout(weights, p=self.dropout_p, training=self.training)
        return torch.matmul(weights, value), weights
