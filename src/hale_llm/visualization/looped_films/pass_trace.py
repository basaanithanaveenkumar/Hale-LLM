"""``PassTrace``: what happened inside the loop of one forward pass, pass by pass."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class PassTrace:
    """Per-pass record of one forward of a looped model on one sequence.

    Attributes:
        text: the input characters.
        states: ``T`` tensors ``(L, d)`` - the hidden state after every pass.
        lens_ids: ``T`` tensors ``(L, k)`` - top-``k`` next-token ids if the model stopped after
            that pass (the *logit lens*: final norm + LM head applied to the intermediate state).
        lens_probs: ``T`` tensors ``(L, k)`` - their probabilities.
        change: ``T`` tensors ``(L,)`` - ``|h_t - h_{t-1}| / |h_{t-1}|`` per token (pass 1: 1).
        halting: ``T`` tensors ``(L,)`` - cumulative ACT halting mass after each pass (ACT only).
        running: ``T`` bool tensors ``(L,)`` - tokens still updating after each pass (ACT / MoR).
        depth: ``(L,)`` passes each token received (ACT / MoR; ``None`` for a plain loop).
        loops: passes the forward actually ran.
    """

    text: str
    states: list[torch.Tensor] = field(default_factory=list)
    lens_ids: list[torch.Tensor] = field(default_factory=list)
    lens_probs: list[torch.Tensor] = field(default_factory=list)
    change: list[torch.Tensor] = field(default_factory=list)
    halting: list[torch.Tensor] = field(default_factory=list)
    running: list[torch.Tensor] = field(default_factory=list)
    depth: torch.Tensor | None = None
    loops: int = 0

    @property
    def n_passes(self) -> int:
        """Number of recorded passes."""
        return len(self.states)

    def residuals(self) -> list[float]:
        """Largest per-token relative change of every pass (the fixed-point residual)."""
        return [float(c.max()) for c in self.change]
