"""Looped LM objective: next-token NLL.

The ACT ponder cost ``tau * mean(N + R)`` is an architecture regulariser, not part of the task:
the trainer adds it for every looped backbone through
:func:`hale_llm.losses.with_auxiliary_losses` (logged as ``train/ponder_loss``), so it is not
added here - that would count it twice.
"""

from __future__ import annotations

import torch

from hale_llm.core.registry import register_loss
from hale_llm.losses import model_token_nll


@register_loss("looped")
def looped_loss(model, batch: dict) -> torch.Tensor:
    logits = model(batch)
    return model_token_nll(
        model,
        logits[:, :-1, :],
        batch["labels"][:, :-1],
        ignore_index=-100,
        reduction="mean",
    )
