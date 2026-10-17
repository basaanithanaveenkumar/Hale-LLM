"""Plain per-token cross-entropy (registered as ``ce``)."""

from __future__ import annotations

import torch

from hale_llm.losses.reduce import per_token_cross_entropy
from hale_llm.losses.registry import register_token_loss


@register_token_loss("ce")
def cross_entropy_token_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    ignore_index: int = -100,
    **_,
) -> torch.Tensor:
    """Unreduced cross-entropy ``(B, L)``; positions equal to ``ignore_index`` give 0."""
    return per_token_cross_entropy(logits, targets, ignore_index=ignore_index)
