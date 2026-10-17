"""Block flow matching objective: masked-token cross-entropy weighted by the path rate.

For a position masked at time ``t`` the discrete-flow-matching loss is
``kappa'(t) / (1 - kappa(t)) * -log p(x_1 | x_t)`` (``1 / (1 - t)`` on the linear path), the
same weighting ``flow_matching`` uses, applied per block with an independent ``t`` per block.
"""

from __future__ import annotations

import torch

from hale_llm.core.registry import register_loss
from hale_llm.losses import model_token_nll
from hale_llm.models.models_block_flow_matching.path import corrupt_blocks


@register_loss("block_flow_matching")
def block_flow_matching_loss(model, batch: dict) -> torch.Tensor:
    """Masked-token NLL weighted by the path rate, with one flow time per block."""
    x1 = batch["input_ids"]
    x_t, is_masked, t = corrupt_blocks(x1, model.block_size, batch["mask_token_id"], model.path)
    logits = model({**batch, "x_clean": x1, "x_t": x_t, "t": t})
    loss_pos = is_masked
    if batch.get("attention_mask") is not None:
        loss_pos = loss_pos & batch["attention_mask"].bool()
    nll = model_token_nll(model, logits, x1, reduction="none")
    weighted = nll * model.path.rate(t) * loss_pos.float()
    return weighted.sum() / loss_pos.float().sum().clamp(min=1.0)
