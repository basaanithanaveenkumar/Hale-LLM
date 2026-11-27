"""One attention head: scaled dot-product attention over a ``head_size`` slice."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from hale_llm.nn.positional.positional_encoding import PositionalEncoding


class HeadAttention(nn.Module):
    """One head of self-attention.

    Shapes: ``B`` batch, ``S`` sequence length, ``D`` model width, ``H`` head size.
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
        # Scale by the width of the vectors being multiplied (head size), not the model width.
        self.scale = 1.0 / math.sqrt(head_size)

        # Linear projections for query, key and value. No bias.
        self.query_proj = nn.Linear(d_model, head_size, bias=False)
        self.key_proj = nn.Linear(d_model, head_size, bias=False)
        self.value_proj = nn.Linear(d_model, head_size, bias=False)

    def forward(
        self,
        inputs: torch.Tensor,
        positions: torch.Tensor | None = None,
        key_padding_mask: torch.Tensor | None = None,
        document_mask: torch.Tensor | None = None,
    ):
        """``inputs`` is ``[B, S, D]``; returns ``[B, S, H]`` (plus the weights if requested).

        ``key_padding_mask`` is ``[B, S]``, ``True`` where a key must be ignored. ``document_mask`` is
        ``[B, S, S]``, ``True`` where query ``i`` must not see key ``j`` (they are in different documents of a
        packed window, see :func:`cross_document_mask`).
        """
        B, S, _ = inputs.shape

        # 1. Project the input to query, key and value.
        query = self.query_proj(inputs)  # [B, S, H]
        key = self.key_proj(inputs)  # [B, S, H]
        value = self.value_proj(inputs)  # [B, S, H]

        # 2. Let the positional encoding act on queries and keys (RoPE rotates them).
        #    It expects a heads axis, so [B, S, H] becomes [B, 1, S, H] and back.
        if self.positional is not None:
            if positions is None:
                positions = torch.arange(S, device=inputs.device)
            query, key = self.positional.encode_queries_keys(
                query.unsqueeze(1), key.unsqueeze(1), positions
            )
            query, key = query.squeeze(1), key.squeeze(1)

        # 3. The fast path: one fused kernel does steps 4-7 below without building [B, S, S].
        #    Plain causal attention lets the kernel make the mask itself.
        if not self.return_attn_weights and key_padding_mask is None and document_mask is None:
            return F.scaled_dot_product_attention(
                query, key, value,
                dropout_p=self.dropout_p if self.training else 0.0,
                is_causal=self.causal,
                scale=self.scale,
            )

        # 3b. Several documents in one window: still a fused kernel, with the mask handed over explicitly
        #     (causal AND same document). The explicit path below does the same job when weights are wanted.
        if not self.return_attn_weights and key_padding_mask is None and document_mask is not None:
            allowed = ~document_mask  # [B, S, S] True = may attend
            if self.causal:
                allowed = allowed & torch.ones(S, S, dtype=torch.bool, device=inputs.device).tril()
            return F.scaled_dot_product_attention(
                query, key, value,
                attn_mask=allowed,
                dropout_p=self.dropout_p if self.training else 0.0,
                scale=self.scale,
            )

        # 4. Score every query against every key.
        scores = torch.matmul(query, key.transpose(-2, -1)) * self.scale  # [B, S, S]

        # 5. Block what must not be seen: the future (causal), padded keys and other documents.
        blocked = torch.zeros(1, S, S, dtype=torch.bool, device=inputs.device)
        if self.causal:
            blocked = blocked | torch.ones(S, S, dtype=torch.bool, device=inputs.device).triu(1)
        if key_padding_mask is not None:
            blocked = blocked | key_padding_mask[:, None, :]  # [B, S, S]
        if document_mask is not None:
            blocked = blocked | document_mask  # [B, S, S]
        # A large finite negative number (not -inf) keeps rows that are all padding from
        # turning into NaN in the softmax.
        scores = scores.masked_fill(blocked, torch.finfo(scores.dtype).min)

        # 6. Softmax over the keys, then dropout on the attention weights.
        attn_weights = F.softmax(scores, dim=-1)  # [B, S, S]
        attn_weights = F.dropout(attn_weights, p=self.dropout_p, training=self.training)

        # 7. Weighted sum of the values.
        attn_output = torch.matmul(attn_weights, value)  # [B, S, H]

        if self.return_attn_weights:
            return attn_output, attn_weights
        return attn_output
