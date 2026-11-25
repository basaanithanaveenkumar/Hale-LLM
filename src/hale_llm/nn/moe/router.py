"""Base class for routers that send each token to its best ``k`` experts."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from hale_llm.nn.moe.router_output import RouterOutput


class Router(nn.Module):
    """Score every expert for every token, keep the top ``k`` and softmax over those.

    Subclasses decide how the scores are made (:meth:`perturb`); the selection and
    the softmax are shared.
    """

    def __init__(self, d_model: int, n_experts: int, top_k: int) -> None:
        super().__init__()
        if not 1 <= top_k <= n_experts:
            raise ValueError(f"top_k={top_k} must be between 1 and n_experts={n_experts}")
        self.top_k = top_k
        self.n_experts = n_experts
        self.gate = nn.Linear(d_model, n_experts)

    def perturb(self, logits: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Scores used to pick experts, derived from the gate logits. Default: unchanged."""
        return logits

    def forward(self, x: torch.Tensor) -> RouterOutput:
        logits = self.gate(x)
        scores = self.perturb(logits, x)
        top_scores, indices = scores.topk(self.top_k, dim=-1)
        # -inf everywhere except the chosen experts, so the softmax gives them all the weight.
        sparse = torch.full_like(scores, float("-inf")).scatter(-1, indices, top_scores)
        return RouterOutput(
            weights=F.softmax(sparse, dim=-1),
            indices=indices,
            probs=F.softmax(logits, dim=-1),
        )
