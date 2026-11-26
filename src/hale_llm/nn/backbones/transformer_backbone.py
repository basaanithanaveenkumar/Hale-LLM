"""Token embedding, decoder stack and (tied) language-model head."""

from __future__ import annotations

import math

import torch
from loguru import logger
from torch import nn

from hale_llm.nn.backbones.decoder_transformer import DecoderTransformer
from hale_llm.nn.blocks import build_ffn
from hale_llm.nn.blocks.residual_init import init_linear_and_embedding, scale_residual_outputs
from hale_llm.nn.positional import build_positional_encoding


class TransformerBackbone(nn.Module):
    """``token ids -> logits`` over the vocabulary.

    The positional encoding comes from the registry (``model.pos_encoding``), the norm
    from ``model.norm_type`` and the feed-forward network from ``model.ffn_type``, so
    switching any of them is a config change.
    """

    def __init__(self, vocab_size: int, model_cfg) -> None:
        super().__init__()
        cfg = model_cfg
        self.vocab_size = vocab_size
        self.max_length = cfg.max_length

        self.positional = build_positional_encoding(
            cfg.pos_encoding,
            d_model=cfg.d_model,
            max_length=cfg.max_length,
            head_dim=cfg.d_model // cfg.n_heads,
            rope_theta=cfg.rope_theta,
            rope_scaling=cfg.rope_scaling,
            rope_scaling_factor=cfg.rope_scaling_factor,
            rope_original_length=cfg.rope_original_length,
        )
        self.token_emb = nn.Embedding(vocab_size, cfg.d_model)
        self.stack = DecoderTransformer(
            n_layers=cfg.n_layers,
            d_model=cfg.d_model,
            n_heads=cfg.n_heads,
            ffn_factory=lambda: build_ffn(cfg.ffn_type, cfg),
            dropout=cfg.dropout,
            norm_type=cfg.norm_type,
            positional=self.positional,
        )
        self.lm_head = nn.Linear(cfg.d_model, vocab_size, bias=False)
        self._init_weights(cfg.n_layers)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.token_emb.weight
        logger.info(
            "TransformerBackbone: layers={} d_model={} heads={} ffn={} positional={} norm={}",
            cfg.n_layers, cfg.d_model, cfg.n_heads, cfg.ffn_type, self.positional.describe(), cfg.norm_type,
        )

    def _init_weights(self, n_layers: int) -> None:
        """GPT-2 style initialisation.

        * Every Linear and Embedding starts at N(0, 0.02), biases at 0. PyTorch's default
          Embedding is N(0, 1); tied to the output head it would give first logits of std
          ~sqrt(d_model) and a first loss near 150 instead of ln(vocab).
        * The layers that write into the residual stream (attention output, feed-forward
          output, expert output) are scaled by 1 / sqrt(2 * n_layers). Each block adds two
          such terms, so without this the stream grows with depth.
        """
        init_linear_and_embedding(self)
        residual_std = 0.02 / math.sqrt(2 * n_layers)
        for layer in self.stack.layers:
            scale_residual_outputs(layer, residual_std)

    def encode(self, token_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        """``token_ids`` is ``[B, S]``; returns the final hidden states ``[B, S, D]`` (after the last norm).

        ``attention_mask`` is 1 for real tokens and 0 for padding.
        """
        B, S = token_ids.shape
        if S > self.max_length:
            raise ValueError(f"sequence of {S} tokens exceeds max_length={self.max_length}")

        # 1. Token ids to vectors, plus a position table if the scheme uses one.
        hidden = self.token_emb(token_ids)  # [B, S, D]
        hidden = self.positional.encode_embeddings(hidden)  # [B, S, D]

        # 2. Which keys are padding, and the position of every token.
        key_padding = (attention_mask == 0) if attention_mask is not None else None  # [B, S]
        positions = torch.arange(S, device=token_ids.device)  # [S]

        # 3. Run the transformer blocks.
        return self.stack(hidden, positions, key_padding)  # [B, S, D]

    def forward(
        self, token_ids: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """``token_ids`` is ``[B, S]``; returns logits ``[B, S, vocab]``."""
        # 4. Project every position to a score for each vocabulary token.
        return self.lm_head(self.encode(token_ids, attention_mask))  # [B, S, vocab]

    def auxiliary_loss(self) -> torch.Tensor:
        """Extra training loss from the layers (MoE load balancing); zero when dense."""
        return self.stack.auxiliary_loss()
