"""MTP objective: the mean over heads of each head's shifted next-``k`` token NLL.

Head ``k`` (0-based) at position ``i`` is scored against ``labels[i + k + 1]``; positions whose
target is padding are ignored. Averaging (rather than summing) keeps the loss on the scale of
an autoregressive NLL, so learning rates transfer between the two variants.
"""

from __future__ import annotations

import torch

from hale_llm.core.registry import register_loss
from hale_llm.losses import model_token_nll


def head_losses(model, batch: dict) -> list[torch.Tensor]:
    """Per-head mean NLL (head 0 = next-token loss)."""
    logits = model(batch)
    labels = batch["labels"]
    out = []
    for k, head_logits in enumerate(logits):
        shift = k + 1
        if labels.size(1) <= shift:
            break
        out.append(
            model_token_nll(
                model,
                head_logits[:, :-shift, :],
                labels[:, shift:],
                ignore_index=batch["pad_token_id"],
                reduction="mean",
            )
        )
    return out


@register_loss("mtp")
def mtp_loss(model, batch: dict) -> torch.Tensor:
    """Mean over heads of each head's shifted next-``k`` NLL (keeps the AR loss scale)."""
    return torch.stack(head_losses(model, batch)).mean()
