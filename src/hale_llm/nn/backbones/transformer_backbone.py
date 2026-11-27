"""The decoder-only transformer: token embedding, transformer blocks, final norm and a (tied) output head."""

from __future__ import annotations

import math

import torch
from loguru import logger
from torch import nn

from hale_llm.nn.attention import cross_document_mask
from hale_llm.nn.blocks import build_blocks
from hale_llm.nn.blocks.residual_init import init_linear_and_embedding, scale_residual_outputs
from hale_llm.nn.norm import build_norm
from hale_llm.nn.positional import build_positional_encoding


class TransformerBackbone(nn.Module):
    """``token ids -> hidden states -> logits`` over the vocabulary.

    Layout (``encode`` returns the hidden states, ``forward`` adds the head)::

        token ids -> token_emb -> [ blocks x n_layers ] -> norm -> hidden states -> lm_head -> logits

    The positional encoding, the norm and the feed-forward network come from registries
    (``model.pos_encoding``, ``model.norm_type``, ``model.ffn_type``), so switching any of them is a
    config change.
    """

    def __init__(self, vocab_size: int, model_cfg) -> None:
        super().__init__()
        cfg = model_cfg
        self.vocab_size = vocab_size
        self.max_length = cfg.max_length

        # 1. Positions: RoPE rotates inside attention, a learned table adds to the embeddings.
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
        # 2. Token ids to vectors.
        self.token_emb = nn.Embedding(vocab_size, cfg.d_model)
        # 3. The transformer blocks, then one final norm before the output head.
        self.layers = build_blocks(cfg, cfg.n_layers, self.positional)
        self.norm = build_norm(cfg.norm_type, cfg.d_model)
        # 4. Vectors to a score per vocabulary token (optionally sharing the embedding's weights).
        self.lm_head = nn.Linear(cfg.d_model, vocab_size, bias=False)

        self._init_weights(cfg.n_layers)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.token_emb.weight
        # Checkpoints from before the blocks moved out of a ``stack`` sub-module still load.
        self._register_load_state_dict_pre_hook(_rename_old_stack_keys)
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
        for block in self.layers:
            scale_residual_outputs(block, residual_std)

    def encode(
        self,
        token_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        document_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """``token_ids`` is ``[B, S]``; returns the final hidden states ``[B, S, D]`` (after the last norm).

        ``attention_mask`` is 1 for real tokens and 0 for padding. ``document_ids`` (``[B, S]``, optional) says which
        document of a packed window each token is in; tokens then only attend within their own document.
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
        document_mask = cross_document_mask(document_ids) if document_ids is not None else None  # [B, S, S]

        # 3. The transformer blocks, each keeping the shape [B, S, D].
        for block in self.layers:
            hidden = block(hidden, positions, key_padding, document_mask)

        # 4. The final norm.
        return self.norm(hidden)  # [B, S, D]

    def forward(
        self,
        token_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        document_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """``token_ids`` is ``[B, S]``; returns logits ``[B, S, vocab]``."""
        # 5. A score for every vocabulary token at every position.
        return self.lm_head(self.encode(token_ids, attention_mask, document_ids))  # [B, S, vocab]


def _rename_old_stack_keys(state_dict, prefix, *_) -> None:
    """Load hook: ``<prefix>stack.layers.N...`` and ``<prefix>stack.norm...`` become ``<prefix>layers.N...``/``norm...``."""
    old = prefix + "stack."
    for key in [k for k in state_dict if k.startswith(old)]:
        state_dict[prefix + key[len(old):]] = state_dict.pop(key)
