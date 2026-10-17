"""One weight-tied pass of the looped model during a generation step."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class LoopPass:
    """State of the slow-motion forward after one pass of the shared block."""

    lens_ids: list[int]  # logit lens top-5 for the next token
    lens_probs: list[float]
    halting: list[float]  # cumulative ACT halting mass per token
    running: list[bool]
    change: list[float]  # |h_t - h_{t-1}| / |h_{t-1}| per token
