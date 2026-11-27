"""Decoder-only transformer with DeepSeek-V3 style multi-token prediction (MTP).

The trunk is the normal :class:`AutoregressiveLM`. ``model.mtp_depth`` extra modules sit on top of it and
are only used for the training loss (and, optionally, as the draft model of speculative decoding).
``forward(batch)`` still returns the trunk's next-token logits, so evaluation, metrics and plain
sampling work exactly as for the other variants.
"""

from __future__ import annotations

import torch

from hale_llm.dataloader.collate import register_collate
from hale_llm.models.models_autoregressive.model import AutoregressiveLM, collate_autoregressive
from hale_llm.models.models_mtp.loss_schedule import MTPLossSchedule
from hale_llm.models.models_mtp.variants import ALL_VARIANTS, MOE_MTP_VARIANT, MTP_VARIANT
from hale_llm.nn.mtp import build_mtp_predictor
from hale_llm.registry import register_variant

for _name in ALL_VARIANTS:
    register_collate(_name)(collate_autoregressive)


@register_variant(MTP_VARIANT)
class MTPAutoregressiveLM(AutoregressiveLM):
    """Trunk + ``model.mtp_depth`` multi-token-prediction depths that share its embedding and output head.

    With ``model.mtp_offset_conditioning`` the depths are replaced by one module that predicts at a random offset.
    """

    def __init__(self, vocab_size: int, cfg) -> None:
        super().__init__(vocab_size, cfg)
        # DeepSeek-V3's chained depths, or one offset-conditioned module (model.mtp_offset_conditioning). The offset order
        # is seeded by model.mtp_offset_seed, or train.seed when that is not set.
        offset_seed = cfg.model.mtp_offset_seed if cfg.model.mtp_offset_seed is not None else cfg.train.seed
        self.mtp = build_mtp_predictor(cfg.model, self.backbone.positional, offset_seed, cfg.train.lean.accum_steps)
        # The MTP blocks may be mixture-of-experts even when the trunk is dense (model.mtp_ffn_type).
        self.uses_moe = self.uses_moe or cfg.model.mtp_block_config().ffn_type == "moe"
        # Lambda follows a schedule; the trainer says how far along the run is (set_training_progress).
        self.loss_schedule = MTPLossSchedule.from_config(cfg.model)
        self.mtp_weight = self.loss_schedule.weight_at(0.0)
        # The pieces of the last training loss ({"ntp_loss": .., "mtp_loss": ..}), detached; the trainer logs them.
        self.loss_parts: dict[str, torch.Tensor] = {}

    def set_training_progress(self, progress: float) -> None:
        """Called by the trainer before every step with ``step / total_steps``: sets the current MTP loss weight."""
        self.mtp_weight = self.loss_schedule.weight_at(progress)

    @staticmethod
    def initial_loss_scale(cfg) -> float:
        """Every head starts near ``ln(vocab)``: the model scores ``ln(vocab) * (1 + first lambda * terms)``, where the
        MTP term adds up ``terms`` such losses (the number of depths / offsets when summed, 1 when averaged)."""
        return 1.0 + cfg.model.mtp_loss_weight * cfg.model.mtp_loss_terms()


@register_variant(MOE_MTP_VARIANT)
class MoEMTPAutoregressiveLM(MTPAutoregressiveLM):
    """The MTP model with a mixture-of-experts trunk. ``RunConfig`` sets ``model.ffn_type: moe`` for it.

    The MTP blocks are mixture-of-experts too unless ``model.mtp_ffn_type`` says otherwise.
    """

    def __init__(self, vocab_size: int, cfg) -> None:
        if cfg.model.ffn_type != "moe":
            raise ValueError(f"variant {MOE_MTP_VARIANT} needs model.ffn_type: moe, got {cfg.model.ffn_type!r}")
        super().__init__(vocab_size, cfg)
