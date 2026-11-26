"""Top-k router with learned Gaussian noise during training."""

import torch
import torch.nn.functional as F
from torch import nn

from hale_llm.nn.moe.registry import register_router
from hale_llm.nn.moe.router import Router
from hale_llm.nn.moe.router_output import RouterOutput


@register_router("noisy_topk")
class NoisyBestKRouter(Router):
    """Adds noise, scaled by a learned amount per token, before picking the experts.

    The noise spreads tokens over the experts early in training so none starves. It is
    only added in training mode; evaluation and generation are deterministic.
    """

    def __init__(self, d_model: int, n_experts: int, top_k: int) -> None:
        super().__init__(d_model, n_experts, top_k)
        self.noise_gate = nn.Linear(d_model, n_experts)

    def forward(self, x: torch.Tensor) -> RouterOutput:
        """``x`` is ``[T, D]`` (T tokens); the weights are ``[T, E]`` for ``E`` experts."""
        # 1. One score per expert for every token.
        #    Routing always runs in fp32: under mixed precision small score differences decide
        #    which expert a token uses, and half precision would blur them.
        logits = self.gate(x).float()  # [T, E]

        # 2. While training, add Gaussian noise whose size the model learns per token and expert.
        noisy_logits = logits
        if self.training:
            noise_scale = F.softplus(self.noise_gate(x).float())  # [T, E]
            noisy_logits = logits + torch.randn_like(logits) * noise_scale

        # 3. Keep the k best experts of every token (chosen on the noisy scores).
        best_logits, indices = noisy_logits.topk(self.top_k, dim=-1)  # both [T, k]

        # 4. Softmax over the chosen experts only: everything else is -inf, so it gets weight 0.
        sparse_logits = torch.full_like(logits, float("-inf")).scatter(-1, indices, best_logits)
        weights = F.softmax(sparse_logits, dim=-1)  # [T, E]

        # 5. Softmax over all experts (without noise), kept for the load-balancing loss.
        probs = F.softmax(logits, dim=-1)  # [T, E]
        return RouterOutput(weights=weights, indices=indices, probs=probs)
