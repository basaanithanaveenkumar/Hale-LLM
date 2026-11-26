"""What a feed-forward layer needs to know about an activation."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ActivationSpec:
    """An activation function and whether it is gated.

    * plain (``gated=False``): ``hidden = fn(w1 x)``, two weight matrices.
    * gated (``gated=True``):  ``hidden = fn(w1 x) * (w3 x)``, a third matrix ``w3`` gates the hidden vector.
    """

    fn: Callable[[torch.Tensor], torch.Tensor]
    gated: bool
