"""Masked-diffusion (MDLM) training loss: corrupt by absorbing-state masking, weight by 1/t."""

from __future__ import annotations

import torch

from hale_llm.core.registry import register_loss
from hale_llm.losses import model_token_nll
from hale_llm.models.models_diffusion.log_linear_schedule import LogLinearSchedule
from hale_llm.models.models_diffusion.noise_schedule import NoiseSchedule


def _corrupt(x0, t, mask_id, schedule: NoiseSchedule | None = None):
    """Mask each token of ``x0`` independently with probability ``schedule.mask_probability(t)``.

    Args:
        x0: ``(B, L)`` clean token ids.
        t: ``(B,)`` diffusion times.
        mask_id: id of the ``<mask>`` token.
        schedule: forward schedule (default :class:`LogLinearSchedule`, P(mask) = t).

    Returns:
        ``(x_t, is_masked)``: the corrupted ids and the boolean mask of corrupted positions.
    """
    schedule = schedule or LogLinearSchedule()
    b, l = x0.shape
    mask_prob = schedule.mask_probability(t)[:, None].expand(b, l)
    is_masked = torch.rand_like(x0, dtype=torch.float) < mask_prob
    x_t = torch.where(is_masked, torch.full_like(x0, mask_id), x0)
    return x_t, is_masked

@register_loss("diffusion")
def diffusion_loss(model, batch: dict) -> torch.Tensor:
    """Continuous-time MDLM loss: schedule-weighted NLL of the masked tokens at a random ``t``."""
    x0 = batch["input_ids"]
    b = x0.size(0)
    device = x0.device
    mask_id = batch["mask_token_id"]
    t = torch.rand(b, device=device) * (1.0 - model.eps) + model.eps
    x_t, is_masked = _corrupt(x0, t, mask_id, model.schedule)
    logits = model({**batch, "x_t": x_t, "t": t})
    loss_pos = is_masked
    if batch.get("attention_mask") is not None:
        loss_pos = loss_pos & batch["attention_mask"].bool()
    nll = model_token_nll(model, logits, x0, reduction="none")
    weight = model.schedule.loss_weight(t, model.eps)[:, None]
    return (nll * weight * loss_pos.float()).sum() / loss_pos.float().sum().clamp(min=1.0)
