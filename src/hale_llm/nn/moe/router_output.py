"""What a router hands to the mixture-of-experts layer."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class RouterOutput:
    """Routing decision for every token.

    Attributes:
        weights: ``(..., n_experts)`` softmax over the chosen experts, zero elsewhere.
        indices: ``(..., k)`` ids of the ``k`` experts picked per token.
        probs: ``(..., n_experts)`` softmax over all experts (used for load balancing).
    """

    weights: torch.Tensor
    indices: torch.Tensor
    probs: torch.Tensor
