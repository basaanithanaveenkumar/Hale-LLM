"""A stack of :class:`DecoderLayer` followed by a final norm."""

from __future__ import annotations

import torch
from hale_core.nn.layers import build_ffn
from torch import nn

from hale_llm.nn.attention import PositionalAttention
from hale_llm.nn.backbones.decoder_layer import DecoderLayer
from hale_llm.nn.norm import build_norm
from hale_llm.nn.positional import PositionalEncoding


class DecoderStack(nn.Module):
    """Hidden states in, hidden states out; no embeddings or output head."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_layers: int,
        d_ff: int,
        n_kv_heads: int,
        positional: PositionalEncoding,
        norm_type: str,
        dropout: float = 0.0,
        ffn_type: str = "mlp",
        moe_num_experts: int = 4,
        moe_top_k: int = 2,
        moe_num_shared: int = 1,
        use_time_cond: bool = False,
    ) -> None:
        super().__init__()
        # Every layer shares the one positional-encoding module (RoPE has no
        # parameters, so sharing costs nothing).
        self.layers = nn.ModuleList(
            DecoderLayer(
                d_model,
                attention=PositionalAttention(d_model, n_heads, n_kv_heads, positional, dropout),
                ffn=build_ffn(
                    ffn_type,
                    d_model=d_model,
                    d_ff=d_ff,
                    dropout=dropout,
                    moe_num_experts=moe_num_experts,
                    moe_top_k=moe_top_k,
                    moe_num_shared=moe_num_shared,
                ),
                norm_type=norm_type,
                use_time_cond=use_time_cond,
            )
            for _ in range(n_layers)
        )
        self.ln_out = build_norm(norm_type, d_model)

    def forward(
        self,
        h: torch.Tensor,
        t_emb: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
        key_padding_mask: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        for layer in self.layers:
            h = layer(
                h,
                t_emb=t_emb,
                attn_mask=attn_mask,
                key_padding_mask=key_padding_mask,
                positions=positions,
            )
        return self.ln_out(h)
