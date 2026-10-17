from __future__ import annotations

import torch
import torch.nn as nn

from hale_llm.core.backbones import build_backbone
from hale_llm.core.registry import register_variant
from hale_llm.dataloader.collate import register_collate


@register_collate("block_diffusion")
def collate_block_diffusion(input_ids: torch.Tensor, tokenizer, **_):
    pad_id = tokenizer.pad_token_id
    return {
        "input_ids": input_ids,
        "attention_mask": (input_ids != pad_id).long(),
        "pad_token_id": pad_id,
        "mask_token_id": tokenizer.mask_token_id,
    }


@register_variant("block_diffusion")
class BlockDiffusionLM(nn.Module):
    def __init__(self, vocab_size: int, cfg):
        super().__init__()
        block_size = cfg.model.block_size or 16
        self.backbone = build_backbone(
            vocab_size, cfg.model, block_size=block_size
        )
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.eps = cfg.sample.eps
        self.steps_per_block = cfg.sample.sampling_steps
        self.cfg = cfg

    def forward(self, batch: dict) -> torch.Tensor:
        x_clean, x_noised = batch["x_clean"], batch["x_noised"]
        t_noised = batch["t_noised"]
        t_clean = torch.zeros_like(t_noised)
        tokens = torch.cat([x_clean, x_noised], dim=1)
        t = torch.cat([t_clean, t_noised], dim=1)
        attn = batch.get("attention_mask")
        if attn is not None:
            attn = torch.cat([attn, attn], dim=1)
        logits = self.backbone(tokens, attention_mask=attn, t=t)
        n = x_clean.size(1)
        return logits[:, n:, :]
