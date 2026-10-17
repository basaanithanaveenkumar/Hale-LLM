"""The diffusion / flow time readout drawn in masked-generation films."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class Clock:
    """How a paradigm's time is drawn in the schedule mini-plot.

    Attributes:
        fn: plotted curve over ``t ∈ [0, 1]``.
        title: what the curve is.
        reverse: time runs 1 → 0 (diffusion) - the axis is drawn right to left so progress
            always moves left to right.
        extra: optional dashed comparison curve and its label.
    """

    fn: Callable[[float], float]
    title: str
    reverse: bool = False
    extra: tuple[Callable[[float], float], str] | None = None
