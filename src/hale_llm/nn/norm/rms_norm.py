"""Root-mean-square layer normalisation (Zhang & Sennrich, 2019)."""

from __future__ import annotations

import torch
from torch import nn

from hale_llm.nn.norm.registry import register_norm


@register_norm("rms")
class RMSNorm(nn.Module):
    """Scales each token vector to unit root-mean-square, then applies a learned gain.

    Tiny example: ``x = [3, 4]`` has ``rms = sqrt((9 + 16) / 2) = 3.54``, so the
    output is ``[0.85, 1.13] * weight``. LayerNorm additionally subtracts the mean
    and learns a bias; RMSNorm skips both, which is cheaper and works as well in
    practice for transformer language models.
    """

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute the statistic in float32 so half-precision training stays stable.
        inv_rms = x.float().pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * inv_rms).type_as(x) * self.weight
