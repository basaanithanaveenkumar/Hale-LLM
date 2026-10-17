"""One denoising step of a masked (diffusion / flow) sampler."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class MaskedStep:
    """One denoising / integration step.

    Attributes:
        t_from / t_to: time before and after the step.
        p_reveal: probability that a masked position is revealed in this step.
        pred: the sampled prediction at every position.
        conf: the model's probability of that prediction at every position.
        revealed: positions revealed in this step.
        x: the canvas after the step.
        block: active block (block variants) or ``None``.
        final_fill: the step is the greedy clean-up pass at the end of a block.
    """

    t_from: float
    t_to: float
    p_reveal: float
    pred: list[int]
    conf: list[float]
    revealed: set[int]
    x: list[int]
    block: int | None = None
    final_fill: bool = False
