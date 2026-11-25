"""One resolved stage of the context schedule."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ContextStagePlan:
    """What training looks like from ``start_step`` until the next stage begins."""

    name: str
    start_step: int
    context_window: int
    rope_theta: float
    batch_size: int
    long_doc_fraction: float
    long_doc_min_tokens: int | None
