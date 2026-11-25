from __future__ import annotations

import torch

from hale_llm.losses import model_token_nll
from hale_llm.models.models_autoregressive.variants import ALL_VARIANTS
from hale_llm.registry import register_loss


def autoregressive_loss(model, batch: dict) -> torch.Tensor:
    """Next-token loss, plus the MoE load-balancing loss while training."""
    logits = model(batch)
    loss = model_token_nll(
        model,
        logits[:, :-1, :],
        batch["labels"][:, :-1],
        ignore_index=-100,
        reduction="mean",
    )
    if model.training and getattr(model, "uses_moe", False):
        loss = loss + model.cfg.model.moe_aux_loss_coef * model.backbone.auxiliary_loss()
    return loss


for _name in ALL_VARIANTS:
    register_loss(_name)(autoregressive_loss)
