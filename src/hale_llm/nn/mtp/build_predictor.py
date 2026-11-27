"""Choose the MTP predictor the config asks for."""

from __future__ import annotations

from torch import nn

from hale_llm.nn.mtp.multi_token_predictor import MultiTokenPredictor
from hale_llm.nn.mtp.offset_conditioned_predictor import OffsetConditionedPredictor
from hale_llm.nn.positional.positional_encoding import PositionalEncoding


def build_mtp_predictor(model_cfg, positional: PositionalEncoding | None, seed: int, accum_steps: int = 1) -> nn.Module:
    """DeepSeek-V3's chained depths by default; one offset-conditioned module if ``model.mtp_offset_conditioning``.

    ``seed`` and ``accum_steps`` (micro-batches per optimizer step) only matter to the offset-conditioned predictor, which
    plans the offsets of a whole step at once.
    """
    if model_cfg.mtp_offset_conditioning:
        return OffsetConditionedPredictor(model_cfg, positional, seed, accum_steps)
    return MultiTokenPredictor(model_cfg, positional)
