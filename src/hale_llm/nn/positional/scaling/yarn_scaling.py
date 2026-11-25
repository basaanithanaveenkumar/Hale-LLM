"""YaRN: Yet another RoPE extensioN (Peng et al., 2023)."""

from __future__ import annotations

import math

import torch

from hale_llm.nn.positional.scaling.registry import register_rope_scaling
from hale_llm.nn.positional.scaling.rope_scaling import RopeScaling


@register_rope_scaling("yarn")
class YarnRopeScaling(RopeScaling):
    """Interpolate only the slow channels, and sharpen attention a little.

    Channels differ in how many times they rotate within the original context ``L``:

    * fast channels (many rotations) already see every angle, so they are left alone;
    * slow channels (less than one rotation) are the ones that would meet unseen angles,
      so they are divided by ``factor`` like linear scaling;
    * channels in between blend the two with a linear ramp ("NTK by parts").

    Stretching also flattens attention; multiplying queries and keys by
    ``0.1 * ln(factor) + 1`` compensates. The formulas match gpt-oss's
    ``RotaryEmbedding`` (``gpt_oss/torch/model.py``).
    """

    def __init__(self, beta_fast: float = 32.0, beta_slow: float = 1.0) -> None:
        self._beta_fast = beta_fast  # channels with more rotations than this keep their frequency
        self._beta_slow = beta_slow  # channels with fewer rotations than this are fully interpolated

    def compute(self, head_dim, theta, factor, original_length) -> tuple[torch.Tensor, float]:
        base = self.base_frequencies(head_dim, theta)
        if factor <= 1.0:
            return base, 1.0

        half = head_dim // 2
        # Channel index at which a channel makes `beta` rotations within the original length.
        def boundary(beta: float) -> float:
            return half * math.log(original_length / (beta * 2 * math.pi)) / math.log(theta)

        low, high = boundary(self._beta_fast), boundary(self._beta_slow)
        if not 0 < low < high < half - 1:
            raise ValueError(
                f"YaRN needs original_length large enough for head_dim={head_dim}, theta={theta:g}: "
                f"channel boundaries {low:.1f}..{high:.1f} must lie inside (0, {half - 1})"
            )
        # ramp = 0 for fast channels (keep), 1 for slow channels (interpolate).
        ramp = ((torch.arange(half, dtype=torch.float32) - low) / (high - low)).clamp(0, 1)
        inv_freq = (base / factor) * ramp + base * (1 - ramp)
        return inv_freq, 0.1 * math.log(factor) + 1.0
