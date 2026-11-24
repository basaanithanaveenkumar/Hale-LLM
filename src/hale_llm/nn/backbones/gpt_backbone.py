"""GPT-style token model with selectable positional encoding and normalisation."""

from __future__ import annotations

import torch
from hale_core.nn.backbones.sequence import SequenceBackbone
from loguru import logger
from torch import nn

from hale_llm.nn.backbones.decoder_stack import DecoderStack
from hale_llm.nn.positional import build_positional_encoding


class GPTBackbone(SequenceBackbone):
    """Token embedding -> position handling -> decoder stack -> tied LM head.

    ``pos_encoding`` and ``norm_type`` are names looked up in the registries, so a
    new scheme only needs a ``@register_*`` decorator, not a change here.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 4,
        d_ff: int = 512,
        max_length: int = 64,
        dropout: float = 0.0,
        attn_type: str = "causal",
        use_time_cond: bool = False,
        block_size: int | None = None,
        tie_embeddings: bool = True,
        attn_impl: str = "mha",
        n_kv_heads: int | None = None,
        ffn_type: str = "mlp",
        moe_num_experts: int = 4,
        moe_top_k: int = 2,
        moe_num_shared: int = 1,
        pos_encoding: str = "rope",
        norm_type: str = "layer",
        rope_theta: float = 10_000.0,
        arch: str = "transformer",
        **_ignored,
    ) -> None:
        # learned_pos=False: the base class would create its own table; ours lives
        # inside the registered positional-encoding module instead.
        super().__init__(
            vocab_size=vocab_size,
            d_model=d_model,
            max_length=max_length,
            attn_type=attn_type,
            block_size=block_size,
            use_time_cond=use_time_cond,
            learned_pos=False,
            tie_embeddings=tie_embeddings,
            arch=arch,
        )
        if d_model % n_heads != 0:
            raise ValueError(f"d_model={d_model} must be divisible by n_heads={n_heads}")

        # The attention flavour decides how many key/value heads exist.
        kv_heads = self._resolve_kv_heads(attn_impl, n_heads, n_kv_heads)

        self.positional = build_positional_encoding(
            pos_encoding,
            d_model=d_model,
            max_length=max_length,
            head_dim=d_model // n_heads,
            rope_theta=rope_theta,
        )
        self.stack = DecoderStack(
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ff=d_ff,
            n_kv_heads=kv_heads,
            positional=self.positional,
            norm_type=norm_type,
            dropout=dropout,
            ffn_type=ffn_type,
            moe_num_experts=moe_num_experts,
            moe_top_k=moe_top_k,
            moe_num_shared=moe_num_shared,
            use_time_cond=use_time_cond,
        )
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self._tie_head(self.lm_head)

        # Names kept for code that reaches into the backbone (viz, summaries).
        self.layers = self.stack.layers
        self.ln_out = self.stack.ln_out
        self.head = self.lm_head

        logger.info(
            "GPTBackbone: positional encoding = {} | norm = {} | attention = {} ({} kv heads)",
            self.positional.describe(),
            norm_type,
            attn_impl,
            kv_heads,
        )

    @staticmethod
    def _resolve_kv_heads(attn_impl: str, n_heads: int, n_kv_heads: int | None) -> int:
        if attn_impl == "mha":
            return n_heads
        if attn_impl == "mqa":
            return 1
        if attn_impl == "gqa":
            return n_kv_heads if n_kv_heads is not None else max(1, n_heads // 4)
        raise KeyError(f"unknown attention impl {attn_impl!r}; use mha, gqa or mqa")

    def embed(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Token embeddings, plus a position table when the scheme uses one."""
        return self.positional.encode_embeddings(self.token_emb(token_ids))
