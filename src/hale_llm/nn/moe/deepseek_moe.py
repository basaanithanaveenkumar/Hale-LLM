"""DeepSeek-style mixture of experts: always-on shared experts plus routed experts."""

from __future__ import annotations

import torch
from torch import nn

from hale_llm.nn.moe.expert import Expert
from hale_llm.nn.moe.registry import build_router


class DeepseekMoE(nn.Module):
    """``output = sum(shared experts)(x) + sum over the top-k of weight * routed_expert(x)``.

    Shared experts see every token and hold common knowledge. The router sends each token
    to its best ``top_k`` routed experts, which can specialise.

    While training, the layer also leaves a load-balancing loss in :attr:`aux_loss`.
    """

    def __init__(
        self,
        d_model: int,
        hidden_dim: int,
        n_routed_experts: int,
        top_k: int,
        n_shared_experts: int = 1,
        router: str = "noisy_topk",
        activation: str = "swiglu",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.router = build_router(router, d_model, n_routed_experts, top_k)
        self.shared_experts = nn.ModuleList(
            Expert(d_model, hidden_dim, dropout, activation) for _ in range(n_shared_experts)
        )
        self.routed_experts = nn.ModuleList(
            Expert(d_model, hidden_dim, dropout, activation) for _ in range(n_routed_experts)
        )
        self.n_routed_experts = n_routed_experts
        # Set on every forward: the balancing loss in training mode, zero in eval mode.
        self.aux_loss: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``x`` is ``[B, S, D]``; ``T = B * S`` tokens, ``E`` routed experts."""
        B, S, D = x.shape

        # 1. Treat every token on its own: [B, S, D] -> [T, D].
        tokens = x.reshape(-1, D)

        # 2. Shared experts: every token goes through every one of them.
        output = torch.zeros_like(tokens)  # [T, D]
        for expert in self.shared_experts:
            output = output + expert(tokens)

        # 3. Routing: which experts each token uses, and how much of each.
        routing = self.router(tokens)  # weights [T, E], indices [T, k], probs [T, E]

        # 4. Routed experts: each one runs once, on just the tokens that picked it.
        for expert_id, expert in enumerate(self.routed_experts):
            picked = (routing.indices == expert_id).any(dim=-1)  # [T] True if this expert was chosen
            token_ids = picked.nonzero(as_tuple=True)[0]  # which tokens
            if token_ids.numel() == 0:
                continue  # nobody chose this expert in this batch
            gate = routing.weights[token_ids, expert_id].unsqueeze(-1)  # [n, 1]
            expert_output = expert(tokens[token_ids]) * gate  # [n, D]
            output.index_add_(0, token_ids, expert_output.to(output.dtype))  # add to those tokens

        # 5. Load-balancing loss (Switch Transformer form): E * sum(share_i * mean_prob_i).
        #    It is 1.0 when tokens are spread evenly and grows when a few experts take most.
        if self.training:
            chosen = torch.zeros_like(routing.probs).scatter(-1, routing.indices, 1.0)  # [T, E]
            share = chosen.mean(dim=0) / routing.indices.size(-1)  # [E] share of all picks
            mean_prob = routing.probs.mean(dim=0)  # [E]
            self.aux_loss = self.n_routed_experts * torch.sum(share * mean_prob)
        else:
            self.aux_loss = tokens.new_zeros(())

        # 6. Back to [B, S, D].
        return output.view(B, S, D)
