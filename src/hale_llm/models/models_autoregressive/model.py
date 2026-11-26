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
    """Decoder-only transformer; ``model.ffn_type`` picks a dense MLP or a mixture of experts."""

    def __init__(self, vocab_size: int, cfg) -> None:
        super().__init__()
        self.backbone = build_backbone(vocab_size, cfg.model)
        self.vocab_size = vocab_size
        self.cfg = cfg
        self.uses_moe = cfg.model.ffn_type == "moe"

    def forward(self, batch: dict) -> torch.Tensor:
        return self.backbone(batch["input_ids"], attention_mask=batch.get("attention_mask"))

    @staticmethod
    def initial_loss_scale(cfg) -> float:
        """A fresh model's loss is about ``ln(vocab)`` times this (the ``--dry-run`` sanity check uses it)."""
        return 1.0


@register_variant(MOE_VARIANT)
class MoEAutoregressiveLM(AutoregressiveLM):
    """Same model with a mixture of experts. ``RunConfig`` sets ``model.ffn_type: moe`` for it."""

    def __init__(self, vocab_size: int, cfg) -> None:
        if cfg.model.ffn_type != "moe":
            raise ValueError(f"variant {MOE_VARIANT} needs model.ffn_type: moe, got {cfg.model.ffn_type!r}")
        super().__init__(vocab_size, cfg)
