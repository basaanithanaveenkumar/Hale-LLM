"""Mean next-token NLL of a looped model's logits (shared by the looped metrics)."""

from __future__ import annotations

import torch

from hale_llm.losses import model_token_nll


def next_token_nll(model, batch: dict, logits: torch.Tensor) -> float:
    """Mean NLL of ``logits`` against the shifted ``batch["labels"]`` (padding ignored).

    Args:
        model: the looped model (provides the loss settings).
        batch: an autoregressive batch with ``labels``.
        logits: ``(B, L, V)`` logits for ``batch["input_ids"]``.

    Returns:
        The mean loss as a Python float.
    """
    return float(
        model_token_nll(
            model, logits[:, :-1, :], batch["labels"][:, :-1], ignore_index=-100, reduction="mean"
        )
    )
