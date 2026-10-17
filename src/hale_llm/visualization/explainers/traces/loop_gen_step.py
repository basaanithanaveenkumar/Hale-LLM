"""One looped generation step: the passes run and the token emitted."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class LoopGenStep:
    top_ids: list[int]
    top_probs: list[float]
    chosen: int
    loops: int  # passes the forward ran
    updates: float  # N(i) of the newest position
