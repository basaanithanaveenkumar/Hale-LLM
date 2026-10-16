"""Deployment adapter of ``variant: mtp`` (shared trunk + K next-token heads)."""

from __future__ import annotations

from functools import cached_property
from typing import Any

import torch
from torch import nn

from hale_llm.deploy.adapters.hale_llm_adapter import HaleLLMAdapter
from hale_llm.deploy.registry import LLM_ADAPTERS


@LLM_ADAPTERS.register_module()
class MultiTokenPredictionLMAdapter(HaleLLMAdapter):
    """``input_ids [B, T] → logits, draft_logits_1 … draft_logits_{K-1}`` (each ``[B, T, V]``).

    One forward yields the next-token distribution plus ``K-1`` draft distributions, which is
    exactly what the self-speculative decoder consumes (draft with heads 1…K-1, verify with
    head 0 on the next forward) - so on-device speculative decoding needs no second model.
    """

    variant = "mtp"

    @cached_property
    def head_names(self) -> tuple[str, ...]:
        k = int(self.run_config.model.n_mtp_heads)
        return ("logits", *(f"draft_logits_{i}" for i in range(1, k)))

    @property
    def outputs(self) -> tuple[str, ...]:  # type: ignore[override] - depends on the config
        return self.head_names

    def signature_inputs(self) -> list[dict[str, Any]]:
        return [self.token_spec("input_ids")]

    def forward_for_export(self, model: nn.Module, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        heads = model({"input_ids": inputs["input_ids"]})
        return dict(zip(self.head_names, heads, strict=True))
