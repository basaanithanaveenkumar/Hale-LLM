"""Sort model parameters by the role they play, so each role gets the right optimizer settings."""

from __future__ import annotations

from collections.abc import Iterable

from torch import nn

# 2-D parameters whose name contains one of these are lookup tables (token
# embeddings, tied output head), not hidden-layer transforms.
_LOOKUP_NAME_HINTS = ("emb", "head")


def is_hidden_matrix(name: str, param: nn.Parameter) -> bool:
    """True for 2-D weights inside the transformer blocks (attention, FFN projections)."""
    return param.ndim == 2 and not any(hint in name for hint in _LOOKUP_NAME_HINTS)


def split_parameters(
    named_params: Iterable[tuple[str, nn.Parameter]],
) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
    """Return ``(hidden_matrices, others)`` for the trainable parameters."""
    hidden, others = [], []
    for name, param in named_params:
        if not param.requires_grad:
            continue
        (hidden if is_hidden_matrix(name, param) else others).append(param)
    return hidden, others
