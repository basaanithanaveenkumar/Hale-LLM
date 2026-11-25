"""Plain RoPE."""

from __future__ import annotations

import torch

from hale_llm.nn.positional.scaling.registry import register_rope_scaling
from hale_llm.nn.positional.scaling.rope_scaling import RopeScaling


@register_rope_scaling("none")
class NoRopeScaling(RopeScaling):
    """Frequencies exactly as trained."""

    def compute(self, head_dim, theta, factor, original_length) -> tuple[torch.Tensor, float]:
        return self.base_frequencies(head_dim, theta), 1.0
