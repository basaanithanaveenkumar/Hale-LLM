"""Deployment adapter of ``variant: block_diffusion`` (AR over blocks, diffusion inside a block)."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from hale_llm.deploy.adapters.hale_llm_adapter import HaleLLMAdapter
from hale_llm.deploy.registry import LLM_ADAPTERS


@LLM_ADAPTERS.register_module()
class BlockDiffusionLMAdapter(HaleLLMAdapter):
    """``x_clean [B, T], x_noised [B, T], t_noised [B, T] → logits [B, T, V]``.

    The two-stream forward of the block-causal model: noised block ``b`` attends to the clean
    blocks before it and to itself. The sampler runs it on the full canvas (``T = max_length``),
    so the sequence dimension is exported statically (``dynamic_seq: false`` by default) and only
    the batch is dynamic; ``T`` must stay a multiple of ``model.block_size``.
    """

    variant = "block_diffusion"
    stream_names = ("x_clean", "x_noised", "t_noised")

    def __init__(self, config: str, dynamic_seq: bool = False, **kwargs: Any) -> None:
        super().__init__(config, dynamic_seq=dynamic_seq, **kwargs)

    def signature_inputs(self) -> list[dict[str, Any]]:
        clean, noised, time = self.stream_names
        return [self.token_spec(clean), self.token_spec(noised),
                {"name": time, "shape": ["batch", "seq"], "dtype": "float32", "value_range": [0.0, 1.0]}]

    def forward_for_export(self, model: nn.Module, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        clean, noised, time = self.stream_names
        return {"logits": model({"x_clean": inputs[clean], "x_noised": inputs[noised], "t_noised": inputs[time]})}

    def calibration_inputs(self, token_ids: torch.Tensor) -> dict[str, torch.Tensor]:
        """One random time per block, tokens of each block masked with that probability."""
        clean, noised, time = self.stream_names
        block = int(self.run_config.model.block_size or 16)
        b, n = token_ids.shape
        t = torch.rand(b, -(-n // block)).clamp(0.05, 0.95).repeat_interleave(block, dim=1)[:, :n]
        masked = torch.rand(b, n) < t
        noisy = torch.where(masked, torch.full_like(token_ids, self.vocab_size - 1), token_ids)
        return {clean: token_ids, noised: noisy, time: t}
