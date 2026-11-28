"""Decoder-only transformer whose middle layers are looped with Parcae's stable injection."""

from __future__ import annotations

import torch
import torch.nn as nn

from hale_llm.dataloader.collate import register_collate
from hale_llm.models.models_autoregressive.model import collate_autoregressive
from hale_llm.models.models_looped.variants import ALL_VARIANTS, LOOPED_VARIANT
from hale_llm.nn.looped import LoopedBackbone
from hale_llm.registry import register_variant

for _name in ALL_VARIANTS:
    register_collate(_name)(collate_autoregressive)


@register_variant(LOOPED_VARIANT)
class LoopedAutoregressiveLM(nn.Module):
    """``prelude -> (stable injection + recurrent blocks) x T -> coda`` (see :class:`LoopedBackbone`).

    The loss, the sampler and the data collation are the plain autoregressive ones: the model still returns next-token
    logits ``[B, S, V]``, so evaluation, metrics and generation work as for every other variant. ``forward`` takes an
    optional ``num_loops`` (in the batch under ``"num_loops"``) to run a different depth than the configured one, e.g. to
    see how the loss changes with more loops at test time.
    """

    def __init__(self, vocab_size: int, cfg) -> None:
        super().__init__()
        self.backbone = LoopedBackbone(vocab_size, cfg.model)
        self.vocab_size = vocab_size
        self.cfg = cfg
        self.uses_moe = cfg.model.ffn_type == "moe"

    def forward(self, batch: dict) -> torch.Tensor:
        return self.backbone(
            batch["input_ids"],
            attention_mask=batch.get("attention_mask"),
            document_ids=batch.get("document_ids"),  # present only when model.document_masking is on
            num_loops=batch.get("num_loops"),
        )

    @staticmethod
    def initial_loss_scale(cfg) -> float:
        """A fresh model's loss is about ``ln(vocab)`` (one next-token head)."""
        return 1.0
