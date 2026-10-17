"""Deployment adapter of ``variant: autoregressive`` (causal LM, one token per forward)."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from hale_llm.deploy.adapters.hale_llm_adapter import HaleLLMAdapter
from hale_llm.deploy.registry import LLM_ADAPTERS


@LLM_ADAPTERS.register_module()
class AutoregressiveLMAdapter(HaleLLMAdapter):
    """``input_ids [B, T] → logits [B, T, V]``.

    The exported graph is the full-sequence (prefill) forward; generation re-runs it on the
    growing prefix, or a serving stack adds its own KV cache (TensorRT-LLM / mlx-lm paths need
    an HF-format model and are not offered for this custom architecture).
    """

    variant = "autoregressive"

    def signature_inputs(self) -> list[dict[str, Any]]:
        """One ``input_ids [batch, seq]`` int64 input."""
        return [self.token_spec("input_ids")]

    def forward_for_export(
        self, model: nn.Module, inputs: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Full-sequence (prefill) forward: ``{'logits': [B, T, V]}``."""
        return {"logits": model({"input_ids": inputs["input_ids"]})}
