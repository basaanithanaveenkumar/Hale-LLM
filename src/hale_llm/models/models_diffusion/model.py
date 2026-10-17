"""Masked-diffusion variant: bidirectional time-conditioned denoiser and its collate function."""

from __future__ import annotations

import torch
import torch.nn as nn

from hale_llm.core.backbones import build_backbone
from hale_llm.core.registry import register_variant
from hale_llm.dataloader.collate import register_collate
from hale_llm.models.models_diffusion.log_linear_schedule import LogLinearSchedule
from hale_llm.models.models_diffusion.noise_schedule import NoiseSchedule


@register_collate("diffusion")
def collate_diffusion(input_ids: torch.Tensor, tokenizer, **_):
    """Batch dict with ``input_ids``, ``attention_mask`` and the pad / mask token ids."""
    pad_id = tokenizer.pad_token_id
    return {
        "input_ids": input_ids,
        "attention_mask": (input_ids != pad_id).long(),
        "pad_token_id": pad_id,
        "mask_token_id": tokenizer.mask_token_id,
    }


@register_variant("diffusion")
class MaskedDiffusionLM(nn.Module):
    """Absorbing-state (MDLM) diffusion LM.

    Attributes:
        schedule: forward-masking :class:`NoiseSchedule` shared by the loss and the sampler.
        eps: smallest diffusion time (avoids the ``1/t`` singularity).
    """
    def __init__(self, vocab_size: int, cfg):
        super().__init__()
        self.backbone = build_backbone(vocab_size, cfg.model)
        self.vocab_size = vocab_size
        self.eps = cfg.sample.eps
        # forward-masking schedule shared by the loss and the sampler (Strategy)
        self.schedule: NoiseSchedule = LogLinearSchedule()
        self.cfg = cfg

    def forward(self, batch: dict) -> torch.Tensor:
        """Logits ``(B, L, V)`` for ``batch['x_t']`` at diffusion times ``batch['t']``."""
        return self.backbone(batch["x_t"], attention_mask=batch.get("attention_mask"), t=batch["t"])
