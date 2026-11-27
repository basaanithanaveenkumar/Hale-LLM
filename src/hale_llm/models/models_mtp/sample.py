"""Text generation for the MTP variant: speculative decoding with the MTP depths, or plain sampling."""

from __future__ import annotations

import torch
from loguru import logger

from hale_llm.models.models_autoregressive.sample import sample_autoregressive
from hale_llm.models.models_mtp.speculative import SpeculativeDecoder
from hale_llm.models.models_mtp.variants import ALL_VARIANTS
from hale_llm.registry import register_sampler


@torch.no_grad()
def sample_mtp(
    model,
    tokenizer,
    prompt_ids: torch.Tensor,
    max_new_tokens: int,
    temperature: float = 1.0,
    return_history: bool = False,
    **kwargs,
):
    """Speculative decoding when ``sample.speculative`` is on and there is one prompt; plain sampling otherwise."""
    if not model.cfg.sample.speculative:
        return sample_autoregressive(
            model, tokenizer, prompt_ids, max_new_tokens, temperature, return_history, **kwargs
        )
    if prompt_ids.size(0) != 1:
        logger.warning(
            "speculative decoding needs a single prompt (got {}); using plain sampling",
            prompt_ids.size(0),
        )
        return sample_autoregressive(
            model, tokenizer, prompt_ids, max_new_tokens, temperature, return_history, **kwargs
        )
    return SpeculativeDecoder(model, tokenizer, temperature).generate(
        prompt_ids, max_new_tokens, return_history
    )


for _name in ALL_VARIANTS:
    register_sampler(_name)(sample_mtp)
