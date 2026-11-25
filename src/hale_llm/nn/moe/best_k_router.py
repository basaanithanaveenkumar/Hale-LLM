"""Plain top-k router."""

import torch
import torch.nn.functional as F

from hale_llm.nn.moe.registry import register_router
from hale_llm.nn.moe.router import Router
from hale_llm.nn.moe.router_output import RouterOutput


@register_router("topk")
class BestKRouter(Router):
    """Send every token to the ``k`` experts with the highest gate score."""

    def forward(self, x: torch.Tensor) -> RouterOutput:
        """``x`` is ``[T, D]`` (T tokens); the weights are ``[T, E]`` for ``E`` experts."""
        # 1. One score per expert for every token.
        logits = self.gate(x)  # [T, E]

        # 2. Keep the k best experts of every token.
        best_logits, indices = logits.topk(self.top_k, dim=-1)  # both [T, k]

        # 3. Softmax over the chosen experts only: everything else is -inf, so it gets weight 0.
        sparse_logits = torch.full_like(logits, float("-inf")).scatter(-1, indices, best_logits)
        weights = F.softmax(sparse_logits, dim=-1)  # [T, E]

        # 4. Softmax over all experts, kept for the load-balancing loss.
        probs = F.softmax(logits, dim=-1)  # [T, E]
        return RouterOutput(weights=weights, indices=indices, probs=probs)
