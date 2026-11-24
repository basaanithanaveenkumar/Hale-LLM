"""Self-attention whose position handling is a pluggable :class:`PositionalEncoding`."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from hale_llm.nn.positional.positional_encoding import PositionalEncoding


class PositionalAttention(nn.Module):
    """Multi-head attention that covers MHA, GQA and MQA with one class.

    The number of key/value heads picks the flavour: ``n_kv_heads == n_heads`` is
    multi-head, ``1`` is multi-query, anything that divides ``n_heads`` is grouped-query.
    Queries and keys pass through the positional encoding before the dot product,
    so rotary embeddings work without this class knowing about them.

    Masks follow the PyTorch convention used by the rest of the repo: ``True`` means
    "blocked".
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: int,
        positional: PositionalEncoding,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model={d_model} must be divisible by n_heads={n_heads}")
        if n_heads % n_kv_heads != 0:
            raise ValueError(f"n_heads={n_heads} must be divisible by n_kv_heads={n_kv_heads}")
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = d_model // n_heads
        self.dropout = dropout
        self.positional = positional
        self.wq = nn.Linear(d_model, n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(d_model, d_model, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        key_padding_mask: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        # 1. Project and split into heads: (batch, heads, seq, head_dim).
        q = self.wq(x).view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(batch, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(batch, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # 2. Inject position information (rotation for RoPE, nothing for learned tables).
        if positions is None:
            positions = torch.arange(seq_len, device=x.device)
        q, k = self.positional.encode_queries_keys(q, k, positions)

        # 3. Share each key/value head across its group of query heads.
        group = self.n_heads // self.n_kv_heads
        if group > 1:
            k = k.repeat_interleave(group, dim=1)
            v = v.repeat_interleave(group, dim=1)

        # 4. Fused attention kernel with an additive mask. A large finite negative
        #    number (not -inf) keeps rows that are fully masked from producing NaN.
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=self._additive_mask(attn_mask, key_padding_mask, q),
            dropout_p=self.dropout if self.training else 0.0,
        )
        out = out.transpose(1, 2).reshape(batch, seq_len, -1)
        return self.wo(out)

    @staticmethod
    def _additive_mask(
        attn_mask: torch.Tensor | None,
        key_padding_mask: torch.Tensor | None,
        like: torch.Tensor,
    ) -> torch.Tensor | None:
        """Merge the two boolean "blocked" masks into one float mask for the kernel."""
        if attn_mask is None and key_padding_mask is None:
            return None
        blocked = torch.zeros(1, 1, 1, 1, dtype=torch.bool, device=like.device)
        if attn_mask is not None:
            blocked = blocked | attn_mask.to(torch.bool)[None, None]
        if key_padding_mask is not None:
            blocked = blocked | key_padding_mask[:, None, None, :]
        return torch.zeros(blocked.shape, dtype=like.dtype, device=like.device).masked_fill(
            blocked, torch.finfo(like.dtype).min
        )
