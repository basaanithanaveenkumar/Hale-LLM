"""Deployment adapter of ``variant: block_flow_matching``."""

from __future__ import annotations

import torch
from torch import nn

from hale_llm.deploy.adapters.block_diffusion_lm_adapter import BlockDiffusionLMAdapter
from hale_llm.deploy.registry import LLM_ADAPTERS


@LLM_ADAPTERS.register_module()
class BlockFlowMatchingLMAdapter(BlockDiffusionLMAdapter):
    """``x_clean [B, T], x_t [B, T], t [B, T] → logits [B, T, V]`` (flow time per noised position).

    Same block-causal two-stream contract as block diffusion; time runs ``0 → 1`` inside each
    block and the clean stream is fed at ``t = 1`` by the model itself.
    """

    variant = "block_flow_matching"
    stream_names = ("x_clean", "x_t", "t")

    def forward_for_export(
        self, model: nn.Module, inputs: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Two-stream forward with flow time ``t``: logits for the ``x_t`` stream."""
        return {
            "logits": model({"x_clean": inputs["x_clean"], "x_t": inputs["x_t"], "t": inputs["t"]})
        }

    def calibration_inputs(self, token_ids: torch.Tensor) -> dict[str, torch.Tensor]:
        """Real text on the block flow path: each token is real with probability ``kappa(t) = t``."""
        sample = super().calibration_inputs(token_ids)
        real = torch.rand(token_ids.shape) < sample["t"]  # kappa(t) = t: real with probability t
        return {
            "x_clean": token_ids,
            "x_t": torch.where(real, token_ids, torch.full_like(token_ids, self.vocab_size - 1)),
            "t": sample["t"],
        }
