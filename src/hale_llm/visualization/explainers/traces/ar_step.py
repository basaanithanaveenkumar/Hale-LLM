"""One autoregressive generation step recorded by the explainer."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ARStep:
    """One decoding step: the top-k next-token distribution, the sampled token, attention."""

    top_ids: list[int]
    top_probs: list[float]
    chosen: int
    chosen_prob: float
    attention: list[float] | None = None  # last layer, head-averaged, query = last position
