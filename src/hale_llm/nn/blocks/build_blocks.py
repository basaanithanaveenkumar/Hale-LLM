"""Build a stack of identical transformer blocks from a model config."""

from __future__ import annotations

from torch import nn

from hale_llm.nn.blocks.ffn_registry import build_ffn
from hale_llm.nn.blocks.transformer_block import TransformerBlock
from hale_llm.nn.positional.positional_encoding import PositionalEncoding


def build_blocks(model_cfg, n_layers: int, positional: PositionalEncoding | None) -> nn.ModuleList:
    """``n_layers`` causal blocks, each with its own attention, norms and feed-forward network.

    Width, heads, dropout, norm and feed-forward type all come from ``model_cfg``, so the trunk and the
    multi-token-prediction depths build their blocks the same way. The blocks share ``positional``.
    """
    return nn.ModuleList(
        TransformerBlock(
            model_cfg.d_model,
            model_cfg.n_heads,
            ffn=build_ffn(model_cfg.ffn_type, model_cfg),
            dropout=model_cfg.dropout,
            causal=True,
            norm_type=model_cfg.norm_type,
            positional=positional,
        )
        for _ in range(n_layers)
    )
