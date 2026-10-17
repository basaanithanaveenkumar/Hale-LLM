"""Discrete flow-matching training loss on the linear masking path."""

from __future__ import annotations

import torch

from hale_llm.core.registry import register_loss
from hale_llm.losses import model_token_nll
from hale_llm.models.models_flow_matching.vector_field import mask_prob


@register_loss("flow_matching")
def flow_matching_loss(model, batch: dict) -> torch.Tensor:
    x1 = batch["input_ids"]
    b, l = x1.shape
    device = x1.device
    t = torch.rand(b, device=device) * (1.0 - model.eps)
    keep_mask = mask_prob(t).clamp(0, 1)[:, None].expand(b, l)
    is_masked = torch.rand(b, l, device=device) < keep_mask
    x_t = torch.where(is_masked, torch.full_like(x1, batch["mask_token_id"]), x1)
    logits = model({**batch, "x_t": x_t, "t": t})
    loss_pos = is_masked
    if batch.get("attention_mask") is not None:
        loss_pos = loss_pos & batch["attention_mask"].bool()
    nll = model_token_nll(model, logits, x1, reduction="none")
    weight = (1.0 / mask_prob(t).clamp(min=model.eps))[:, None]
    return (nll * weight * loss_pos.float()).sum() / loss_pos.float().sum().clamp(min=1.0)
