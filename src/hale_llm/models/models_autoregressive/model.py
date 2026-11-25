"""Autoregressive language models: a dense transformer and a mixture-of-experts one.

Every variant follows the same pattern:

  model.py     ``@register_variant``  wraps the backbone and defines ``forward(batch)``
  loss.py      ``@register_loss``     ``(model, batch) -> scalar``
  sample.py    ``@register_sampler``  text generation
  collate      registered here, so the data package never imports a variant by name
"""

from __future__ import annotations

import torch
import torch.nn as nn
from loguru import logger

from hale_llm.dataloader.collate import register_collate
from hale_llm.dataloader.padding import pad_positions
from hale_llm.models.models_autoregressive.variants import ALL_VARIANTS, DENSE_VARIANT, MOE_VARIANT
from hale_llm.nn.backbones import build_backbone
from hale_llm.registry import register_variant


def collate_autoregressive(input_ids: torch.Tensor, tokenizer, **_):
    """Shifted next-token targets. Pad positions are ignored in the loss via labels=-100."""
    pad_id = tokenizer.pad_token_id
    pad = pad_positions(input_ids, pad_id)  # all False for packed windows (no pad token)
    labels = input_ids.clone()
    labels[:, :-1] = input_ids[:, 1:]
    labels[:, -1] = -100
    labels[pad] = -100
    return {
        "input_ids": input_ids,
        "attention_mask": (~pad).long(),
        "labels": labels,
        "pad_token_id": pad_id,
    }


for _name in ALL_VARIANTS:
    register_collate(_name)(collate_autoregressive)


@register_variant(DENSE_VARIANT)
class AutoregressiveLM(nn.Module):
    """Decoder-only transformer. The feed-forward type comes from ``model.ffn_type``."""

    #: Feed-forward type this variant forces; ``None`` = read ``model.ffn_type``.
    forced_ffn_type: str | None = None

    def __init__(self, vocab_size: int, cfg) -> None:
        super().__init__()
        ffn_type = self.forced_ffn_type or cfg.model.ffn_type
        if self.forced_ffn_type and cfg.model.ffn_type != self.forced_ffn_type:
            logger.info(
                "variant {} uses ffn_type={} (model.ffn_type={} is ignored)",
                self.variant_name, self.forced_ffn_type, cfg.model.ffn_type,
            )
        self.backbone = build_backbone(vocab_size, cfg.model, ffn_type=ffn_type)
        self.vocab_size = vocab_size
        self.cfg = cfg
        self.uses_moe = ffn_type == "moe"

    def forward(self, batch: dict) -> torch.Tensor:
        return self.backbone(batch["input_ids"], attention_mask=batch.get("attention_mask"))


@register_variant(MOE_VARIANT)
class MoEAutoregressiveLM(AutoregressiveLM):
    """The same transformer with a DeepSeek-style mixture of experts in every block."""

    forced_ffn_type = "moe"
