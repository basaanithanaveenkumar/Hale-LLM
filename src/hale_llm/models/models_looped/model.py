"""Looped LM: a causal language model whose depth is a loop over one shared block.

``h_t = core(inject(h_{t-1}, e) + E^t)`` for ``t = 1..T`` between an optional prelude and coda
(:class:`hale_core.nn.stacks.LoopedStack`). Parameters are those of ``prelude + core + coda``;
compute is ``prelude + T * core + coda`` and ``T`` can change at inference (test-time compute)
or per token (ACT). The paradigm is next-token prediction, so batches, labels and decoding
match the autoregressive variant; what differs is the backbone, its auxiliary ponder loss and
the loop-count controls exposed here.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from hale_core.nn.stacks import LoopSpec, LoopStats, loop_budget

from hale_llm.core.backbones import build_backbone
from hale_llm.core.registry import register_variant
from hale_llm.dataloader.collate import register_collate


@register_collate("looped")
def collate_looped(input_ids: torch.Tensor, tokenizer, **_) -> dict:
    """Shifted next-token targets; pad positions get ``labels = -100``."""
    pad_id = tokenizer.pad_token_id
    labels = input_ids.clone()
    labels[:, :-1] = input_ids[:, 1:]
    labels[:, -1] = -100
    labels[input_ids == pad_id] = -100
    return {
        "input_ids": input_ids,
        "attention_mask": (input_ids != pad_id).long(),
        "labels": labels,
        "pad_token_id": pad_id,
    }


@register_variant("looped")
class LoopedLM(nn.Module):
    """Causal LM on a :class:`~hale_core.nn.backbones.LoopedBackbone`.

    ``model.arch`` is forced to ``looped`` (the YAML may leave the base default); ``model.loop``
    configures the iteration. ``model.attn_type`` must be ``causal``.

    Args:
        vocab_size: tokenizer size.
        cfg: run config (``cfg.model`` sizes the backbone, ``cfg.model.loop`` the loop).
    """

    def __init__(self, vocab_size: int, cfg) -> None:
        super().__init__()
        if cfg.model.attn_type != "causal":
            raise ValueError(
                f"variant 'looped' is a causal LM; got attn_type={cfg.model.attn_type!r} "
                "(use arch: looped under a diffusion variant for bidirectional loops)"
            )
        self.backbone = build_backbone(vocab_size, cfg.model, arch="looped")
        self.vocab_size = vocab_size
        self.cfg = cfg

    @property
    def loop(self) -> LoopSpec:
        """The loop configuration of the backbone."""
        return self.backbone.loop

    @property
    def last_loop_stats(self) -> LoopStats | None:
        """What the last forward did (passes run, ACT ponder cost / mean updates)."""
        return self.backbone.stack.last_stats

    def forward(self, batch: dict) -> torch.Tensor:
        """``(batch, seq, vocab)`` next-token logits at the current loop budget."""
        return self.backbone(batch["input_ids"], attention_mask=batch.get("attention_mask"))

    def forward_with_loops(self, batch: dict, n_loops: int) -> torch.Tensor:
        """Logits with ``n_loops`` passes (applies outside training; training samples / uses T)."""
        with loop_budget(self, n_loops):
            return self(batch)
