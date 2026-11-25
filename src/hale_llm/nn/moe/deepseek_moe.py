"""DeepSeek-style mixture of experts: always-on shared experts plus routed experts."""

from __future__ import annotations

import torch
from torch import nn

from hale_llm.nn.moe.expert import Expert
from hale_llm.nn.moe.registry import build_router
from hale_llm.nn.moe.router_output import RouterOutput


class DeepseekMoE(nn.Module):
    """``y = sum(shared experts)(x) + sum_k weight_k * routed_expert_k(x)``.

    Shared experts see every token and hold common knowledge; the router sends each
    token to its best ``top_k`` routed experts, which can specialise.

    During training the layer also computes a load-balancing loss (Switch Transformer
    form, ``n_experts * sum(fraction_i * mean_prob_i)``). It is 1.0 when tokens are
    spread evenly and grows when a few experts take everything. Read it with
    :attr:`aux_loss`.
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
        self.aux_loss: torch.Tensor = torch.zeros(())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, dim = x.shape
        flat = x.reshape(-1, dim)  # (tokens, d_model)

        output = torch.zeros_like(flat)
        for expert in self.shared_experts:
            output = output + expert(flat)

        routing = self.router(flat)
        output = self._add_routed(flat, output, routing)
        self.aux_loss = self._load_balance_loss(routing) if self.training else flat.new_zeros(())
        return output.view(batch, seq_len, dim)

    def _add_routed(
        self, flat: torch.Tensor, output: torch.Tensor, routing: RouterOutput
    ) -> torch.Tensor:
        """Run each routed expert only on the tokens that chose it."""
        for expert_id, expert in enumerate(self.routed_experts):
            token_ids = (routing.indices == expert_id).any(dim=-1).nonzero(as_tuple=True)[0]
            if token_ids.numel() == 0:
                continue
            gate = routing.weights[token_ids, expert_id].unsqueeze(-1)
            update = (expert(flat[token_ids]) * gate).to(output.dtype)
            output = output.index_add(0, token_ids, update)
        return output

    def _load_balance_loss(self, routing: RouterOutput) -> torch.Tensor:
        """Penalise routing that piles tokens onto few experts."""
        chosen = torch.zeros_like(routing.probs).scatter(-1, routing.indices, 1.0)
        fraction = chosen.mean(dim=0) / routing.indices.size(-1)  # share of picks per expert
        mean_prob = routing.probs.mean(dim=0)
        return self.n_routed_experts * torch.sum(fraction * mean_prob)
