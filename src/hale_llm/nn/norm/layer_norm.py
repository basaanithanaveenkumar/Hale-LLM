"""LayerNorm under the same registry as RMSNorm."""

from __future__ import annotations

from torch import nn

from hale_llm.nn.norm.registry import register_norm


@register_norm("layer")
class LayerNorm(nn.LayerNorm):
    """Standard ``torch.nn.LayerNorm`` taking only the feature size."""

    def __init__(self, dim: int) -> None:
        super().__init__(dim)
