"""Block flow matching LM: two token streams, block-causal attention, per-position flow time.

The backbone sees ``[x_clean ; x_t]`` (doubled length) under the ``block_causal`` mask: noised
block ``b`` attends to clean blocks ``< b`` and to itself, so a single forward trains every block
in parallel, each conditioned on its clean prefix (the block-diffusion recipe). Time conditions
every position: clean tokens sit at the end of the flow (``t = 1``), noised tokens at their
block's ``t``. Logits are read from the noised stream.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from hale_llm.core.backbones import build_backbone
from hale_llm.core.registry import register_variant
from hale_llm.dataloader.collate import register_collate
from hale_llm.exceptions import ModelConfigurationError
from hale_llm.models.models_block_flow_matching.path import MaskingPath


@register_collate("block_flow_matching")
def collate_block_flow_matching(input_ids: torch.Tensor, tokenizer, **_) -> dict:
    """Clean targets ``x_1`` plus the ids the corruption and the loss need."""
    pad_id = tokenizer.pad_token_id
    return {
        "input_ids": input_ids,
        "attention_mask": (input_ids != pad_id).long(),
        "pad_token_id": pad_id,
        "mask_token_id": tokenizer.mask_token_id,
    }


@register_variant("block_flow_matching")
class BlockFlowMatchingLM(nn.Module):
    """Denoiser ``p(x_1 | x_t, clean prefix, t)`` for block flow matching.

    Config: ``model.block_size`` (default 16, must divide ``max_length``), ``model.attn_type``
    must be ``block_causal`` and ``model.use_time_cond`` true; ``model.fm_schedule`` picks the
    ``kappa`` schedule; ``sample.sampling_steps`` is the Euler steps per block and
    ``sample.eps`` the distance kept from ``t = 1``.
    """

    def __init__(self, vocab_size: int, cfg) -> None:
        super().__init__()
        if cfg.model.attn_type != "block_causal":
            raise ModelConfigurationError(
                f"block_flow_matching needs attn_type: block_causal, got {cfg.model.attn_type!r}"
            )
        if not cfg.model.use_time_cond:
            raise ModelConfigurationError(
                "block_flow_matching conditions on the flow time: set use_time_cond"
            )
        block_size = cfg.model.block_size or 16
        if cfg.model.max_length % block_size != 0:
            raise ModelConfigurationError(
                f"max_length {cfg.model.max_length} must be divisible by block_size {block_size}"
            )
        self.backbone = build_backbone(vocab_size, cfg.model, block_size=block_size)
        self.path = MaskingPath(cfg.model.fm_schedule, cfg.sample.eps)
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.eps = cfg.sample.eps
        self.steps_per_block = cfg.sample.sampling_steps
        self.cfg = cfg

    def forward(self, batch: dict) -> torch.Tensor:
        """Logits ``(batch, seq, vocab)`` of the noised stream.

        Batch keys: ``x_clean`` (conditioning tokens), ``x_t`` (noised tokens), ``t``
        (``(batch, seq)`` flow time of each noised position), optional ``attention_mask``.
        """
        x_clean, x_t, t_noised = batch["x_clean"], batch["x_t"], batch["t"]
        tokens = torch.cat([x_clean, x_t], dim=1)
        t = torch.cat([torch.ones_like(t_noised), t_noised], dim=1)
        attn = batch.get("attention_mask")
        if attn is not None:
            attn = torch.cat([attn, attn], dim=1)
        logits = self.backbone(tokens, attention_mask=attn, t=t)
        return logits[:, x_clean.size(1) :, :]
