"""Discrete flow-matching variant and its collate function."""

from __future__ import annotations

import torch
import torch.nn as nn

from hale_llm.core.backbones import build_backbone
from hale_llm.core.registry import register_variant
from hale_llm.dataloader.collate import register_collate


@register_collate("flow_matching")
def collate_flow_matching(input_ids: torch.Tensor, tokenizer, **_):
    """Batch dict with clean ``input_ids``, ``attention_mask`` and the pad / mask token ids."""
    pad_id = tokenizer.pad_token_id
    return {
        "input_ids": input_ids,  # x1 clean
        "attention_mask": (input_ids != pad_id).long(),
        "pad_token_id": pad_id,
        "mask_token_id": tokenizer.mask_token_id,
    }


@register_variant("flow_matching")
class FlowMatchingLM(nn.Module):
    """Bidirectional, time-conditioned posterior ``p(x_1 | x_t, t)`` on the linear masking path."""

    def __init__(self, vocab_size: int, cfg):
        super().__init__()
        self.backbone = build_backbone(vocab_size, cfg.model)
        self.vocab_size = vocab_size
        self.eps = cfg.sample.eps
        self.cfg = cfg

    def forward(self, batch: dict) -> torch.Tensor:
        """Logits ``(B, L, V)`` for ``batch['x_t']`` at flow times ``batch['t']``."""
        return self.backbone(batch["x_t"], attention_mask=batch.get("attention_mask"), t=batch["t"])
