"""How many times each sequence loops during training."""

from __future__ import annotations

import math

import torch

SAMPLING_MODES = ("poisson", "fixed")


class RecurrenceSampler:
    """Draws the number of loops ``T`` of every sequence, and says how many of the last loops get a gradient.

    Parcae (Sec. 2.1, 4.2 and Algorithm 2/4) samples the depth ``T ~ Lambda(mu_rec)`` *per sequence* instead of per batch,
    which reduces loss spikes, and it parameterises ``Lambda`` by ``mu_rec`` alone: the backward pass is truncated to the
    last ``mu_bwd`` loops *after* ``T`` was drawn (it does not shift the distribution of ``T``). ``mu_bwd = ceil(mu_rec / 2)``.

    * ``"poisson"``: ``T ~ Poisson(mu_rec)``, then clamped to at least 1 (a draw of 0 would skip the recurrent blocks; the
      paper does not say how it avoids that, it has probability ``exp(-mu_rec)``, about 3e-4 for 8).
    * ``"fixed"``: ``T = mu_rec`` for everyone (for tests and ablations).
    """

    def __init__(self, mean_recurrence: int, mean_backprop: int | None = None, mode: str = "poisson") -> None:
        if mean_recurrence < 1:
            raise ValueError(f"mean_recurrence={mean_recurrence} must be at least 1")
        if mode not in SAMPLING_MODES:
            raise ValueError(f"mode={mode!r} must be one of {SAMPLING_MODES}")
        self.mean_recurrence = mean_recurrence
        self.mean_backprop = mean_backprop if mean_backprop is not None else math.ceil(mean_recurrence / 2)
        if self.mean_backprop < 1:
            raise ValueError(f"mean_backprop={self.mean_backprop} must be at least 1")
        self.mode = mode

    def sample(self, batch_size: int, device: torch.device | str = "cpu") -> torch.Tensor:
        """``T`` for each of ``batch_size`` sequences, ``[B]`` int64, each at least 1."""
        if self.mode == "fixed":
            depths = torch.full((batch_size,), self.mean_recurrence, dtype=torch.long)
        else:
            # Drawn on the CPU with the global generator (so torch.manual_seed controls it), then moved.
            rates = torch.full((batch_size,), float(self.mean_recurrence))
            depths = torch.poisson(rates).long().clamp_min(1)
        return depths.to(device)
