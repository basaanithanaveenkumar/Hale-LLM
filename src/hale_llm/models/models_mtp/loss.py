"""Training loss of the MTP variant: next-token loss plus the weighted multi-token losses."""

from __future__ import annotations

import torch

from hale_llm.losses import model_token_nll
from hale_llm.models.models_mtp.variants import ALL_VARIANTS
from hale_llm.nn.moe import auxiliary_loss
from hale_llm.registry import register_loss


def mtp_loss(model, batch: dict) -> torch.Tensor:
    """``next_token + lambda * mean_k(loss_k)`` while training; the next-token loss alone in eval.

    ``lambda`` is ``model.mtp_weight``, set by the trainer from the schedule (0.3, then 0.1 by default).

    Evaluation returns only the next-token loss, so ``val/loss`` is comparable with the other variants
    (the same choice the MoE variant makes for its balancing loss).
    """
    cfg = model.cfg.model
    backbone = model.backbone
    ids, mask, labels = batch["input_ids"], batch.get("attention_mask"), batch["labels"]

    # 1. Trunk: hidden states [B, T, D], logits [B, T, V], and the usual next-token loss.
    #    labels[i] is the token after position i (-100 where there is nothing to predict).
    hidden = backbone.encode(ids, attention_mask=mask)
    trunk_logits = backbone.lm_head(hidden)
    ntp_loss = model_token_nll(
        model, trunk_logits[:, :-1], labels[:, :-1], ignore_index=-100, reduction="mean"
    )
    if not model.training:
        return ntp_loss

    # 2. MTP depths. Depth k returns [B, T-k, V]; its position j predicts t_{j+k+1} = labels[j+k].
    depth_logits = model.mtp(hidden, ids, mask, backbone.token_emb, backbone.lm_head)
    depth_losses = []
    for k, logits in enumerate(depth_logits, start=1):
        # Drop the last position (no target): [B, T-k-1, V] against labels[k .. T-2] = t_{k+1 .. T-1}.
        depth_losses.append(
            model_token_nll(
                model, logits[:, :-1], labels[:, k:-1], ignore_index=-100, reduction="mean"
            )
        )
    mtp_loss_mean = torch.stack(depth_losses).mean()  # (1/D) * sum_k loss_k

    # 3. Combine with the current lambda. lambda * mean_k == (lambda / D) * sum_k.
    loss = ntp_loss + model.mtp_weight * mtp_loss_mean
    if model.uses_moe:
        loss = loss + cfg.moe_aux_loss_coef * auxiliary_loss(model)

    # 4. Remember the pieces for logging (detached: no extra graph).
    model.loss_parts = {
        "ntp_loss": ntp_loss.detach(),
        "mtp_loss": mtp_loss_mean.detach(),
        "mtp_weight": torch.tensor(model.mtp_weight, device=ntp_loss.device),
    }
    for k, depth_loss in enumerate(depth_losses, start=1):
        model.loss_parts[f"mtp_loss_depth{k}"] = depth_loss.detach()
    return loss


for _name in ALL_VARIANTS:
    register_loss(_name)(mtp_loss)
