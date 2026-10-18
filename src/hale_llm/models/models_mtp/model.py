"""Multi-token prediction (MTP) LM: one shared trunk, K heads predicting tokens t+1 ... t+K.

Gloeckle et al., *Better & Faster Large Language Models via Multi-token Prediction*, 2024
(arXiv:2404.19737): the trunk's hidden state at position ``i`` feeds ``K`` heads; head ``k``
predicts token ``i + k + 1``. Head 0 is the ordinary next-token head (the backbone's LM head);
heads ``1..K-1`` are a residual MLP each followed by the **shared** unembedding. Training sums
the next-``k`` losses, which pushes the trunk to plan ahead; at inference the extra heads draft
tokens that the main head verifies in a single forward (self-speculative decoding, see
``sample.py``).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from hale_llm.core.backbones import build_backbone
from hale_llm.core.registry import register_variant
from hale_llm.dataloader.collate import register_collate
from hale_llm.exceptions import ModelConfigurationError
from hale_llm.models.models_mtp.mtp_head import MTPHead


@register_collate("mtp")
def collate_mtp(input_ids: torch.Tensor, tokenizer, **_) -> dict:
    """Unshifted ids as ``labels``: head ``k`` reads ``labels[:, k + 1:]``; pads are ignored."""
    pad_id = tokenizer.pad_token_id
    return {
        "input_ids": input_ids,
        "attention_mask": (input_ids != pad_id).long(),
        "pad_token_id": pad_id,
        "labels": input_ids,
    }


@register_variant("mtp")
class MultiTokenPredictionLM(nn.Module):
    """Causal trunk + ``model.n_mtp_heads`` prediction heads (head 0 = next token).

    ``forward`` returns a list of ``K`` logits tensors ``(batch, seq, vocab)``; entry ``k`` at
    position ``i`` is the distribution of token ``i + k + 1``.
    """

    def __init__(self, vocab_size: int, cfg) -> None:
        super().__init__()
        if cfg.model.n_mtp_heads < 1:
            raise ModelConfigurationError(f"n_mtp_heads must be >= 1, got {cfg.model.n_mtp_heads}")
        if cfg.model.attn_type != "causal":
            raise ModelConfigurationError(
                f"mtp is a causal LM; got attn_type={cfg.model.attn_type!r}"
            )
        self.n_heads = cfg.model.n_mtp_heads
        # Untied: the shared unembedding serves K heads (hale-mtp loads it with the same flag).
        self.backbone = build_backbone(vocab_size, cfg.model, tie_embeddings=False)
        self.extra_heads = nn.ModuleList(
            [
                MTPHead(cfg.model.d_model, cfg.model.d_ff, cfg.model.dropout)
                for _ in range(self.n_heads - 1)
            ]
        )
        self.vocab_size = vocab_size
        self.cfg = cfg

    def trunk(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Final hidden states ``(batch, seq, d_model)`` of the shared backbone."""
        bb = self.backbone
        attn_mask, key_padding = bb.attention_inputs(input_ids, attention_mask, None, None)
        positions = bb.positions(input_ids.size(1), input_ids.device)
        return bb.encode(bb.embed(input_ids), None, attn_mask, key_padding, positions)

    def forward(self, batch: dict) -> list[torch.Tensor]:
        """K logits tensors ``(B, L, V)``: head 0 is next-token, head ``k`` predicts ``i + k +
        1``.
        """
        hidden = self.trunk(batch["input_ids"], batch.get("attention_mask"))
        head = self.backbone.lm_head
        return [head(hidden)] + [head(extra(hidden)) for extra in self.extra_heads]
