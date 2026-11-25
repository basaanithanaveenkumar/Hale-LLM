"""Top-k router with learned Gaussian noise during training."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from hale_llm.nn.moe.registry import register_router
from hale_llm.nn.moe.router import Router


@register_router("noisy_topk")
class NoisyBestKRouter(Router):
    """Adds noise scaled by a learned, token-dependent amount before picking experts.

    The noise spreads tokens over experts early in training so no expert starves.
    It is only added in training mode; evaluation and generation are deterministic.
    """

    def __init__(self, d_model: int, n_experts: int, top_k: int) -> None:
        super().__init__(d_model, n_experts, top_k)
        self.noise_gate = nn.Linear(d_model, n_experts)

    def perturb(self, logits: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return logits
        return logits + torch.randn_like(logits) * F.softplus(self.noise_gate(x))
