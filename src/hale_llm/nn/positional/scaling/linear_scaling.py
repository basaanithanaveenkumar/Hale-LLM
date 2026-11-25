"""Position interpolation (Chen et al., 2023)."""

from __future__ import annotations

import torch

from hale_llm.nn.positional.scaling.registry import register_rope_scaling
from hale_llm.nn.positional.scaling.rope_scaling import RopeScaling


@register_rope_scaling("linear")
class LinearRopeScaling(RopeScaling):
    """Squeeze positions by ``factor``: position 8000 behaves like position 2000 when factor is 4.

    Simple, but it blurs the fine distinctions between neighbouring tokens that the
    high-frequency channels encode, which is what YaRN avoids.
    """

    def compute(self, head_dim, theta, factor, original_length) -> tuple[torch.Tensor, float]:
        return self.base_frequencies(head_dim, theta) / factor, 1.0
