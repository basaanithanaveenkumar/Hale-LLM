"""Base class of the routers that send each token to its best ``k`` experts."""

from __future__ import annotations

from torch import nn


class Router(nn.Module):
    """Holds the gate (one score per expert) and checks ``top_k``.

    The two routers, :class:`BestKRouter` and :class:`NoisyBestKRouter`, each write out
    their whole ``forward`` so the routing can be read top to bottom.
    """

    def __init__(self, d_model: int, n_experts: int, top_k: int) -> None:
        super().__init__()
        if not 1 <= top_k <= n_experts:
            raise ValueError(f"top_k={top_k} must be between 1 and n_experts={n_experts}")
        self.top_k = top_k
        self.n_experts = n_experts
        self.gate = nn.Linear(d_model, n_experts)
