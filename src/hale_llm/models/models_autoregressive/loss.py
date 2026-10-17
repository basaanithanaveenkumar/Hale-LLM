"""Next-token objective of the autoregressive reference variant."""

from __future__ import annotations

import torch

from hale_llm.core.registry import register_loss
from hale_llm.losses import model_token_nll


@register_loss("autoregressive")
def autoregressive_loss(model, batch: dict) -> torch.Tensor:
    """Mean next-token NLL of ``model(batch)`` against the shifted ``labels`` (padding ignored)."""
    logits = model(batch)
    return model_token_nll(
        model,
        logits[:, :-1, :],
        batch["labels"][:, :-1],
        ignore_index=-100,
        reduction="mean",
    )
