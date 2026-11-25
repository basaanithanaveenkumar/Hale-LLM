"""Sort model parameters by the role they play, so each role gets the right optimizer settings."""

from __future__ import annotations

from collections.abc import Iterable

from torch import nn

# 2-D parameters with one of these as a name segment are not plain hidden-layer
# transforms: lookup tables (token embeddings, tied output head, learned positions) and
# the MoE routers, which are trained with AdamW. Whole segments are matched, because the
# attention heads live under ``heads.<i>`` and must stay on Muon.
_NON_HIDDEN_SEGMENTS = frozenset({"token_emb", "lm_head", "positional", "gate", "noise_gate"})


def is_hidden_matrix(name: str, param: nn.Parameter) -> bool:
    """True for 2-D weights inside the transformer blocks (attention and FFN projections)."""
    return param.ndim == 2 and _NON_HIDDEN_SEGMENTS.isdisjoint(name.split("."))


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
