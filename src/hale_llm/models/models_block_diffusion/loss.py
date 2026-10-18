"""Block-diffusion training loss: one noise level per block, clean context through the other
stream.
"""

from __future__ import annotations

import torch

from hale_llm.core.registry import register_loss
from hale_llm.exceptions import ModelConfigurationError
from hale_llm.losses import model_token_nll


@register_loss("block_diffusion")
def block_diffusion_loss(model, batch: dict) -> torch.Tensor:
    """1/t-weighted masked-token NLL with one diffusion time per block.

    Clean blocks feed the second stream, so noised block ``b`` sees the clean blocks before it.

    Raises:
        ModelConfigurationError: if the sequence length is not a multiple of ``block_size``.
    """
    x = batch["input_ids"]
    b, n = x.shape
    bs = model.block_size
    if n % bs != 0:
        raise ModelConfigurationError(f"seq_len {n} must be divisible by block_size {bs}")
    num_blocks = n // bs
    t_block = torch.rand(b, num_blocks, device=x.device) * (1.0 - model.eps) + model.eps
    t_noised = t_block.repeat_interleave(bs, dim=1)
    is_masked = torch.rand(b, n, device=x.device) < t_noised
    x_t = torch.where(is_masked, torch.full_like(x, batch["mask_token_id"]), x)
    logits = model({**batch, "x_clean": x, "x_noised": x_t, "t_noised": t_noised})
    loss_pos = is_masked
    if batch.get("attention_mask") is not None:
        loss_pos = loss_pos & batch["attention_mask"].bool()
    nll = model_token_nll(model, logits, x, reduction="none")
    weight = 1.0 / t_noised.clamp(min=model.eps)
    return (nll * weight * loss_pos.float()).sum() / loss_pos.float().sum().clamp(min=1.0)
