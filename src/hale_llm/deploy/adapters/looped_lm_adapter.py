"""Deployment adapter of ``variant: looped`` (weight-tied recurrent-depth transformer)."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from hale_llm.deploy.adapters.hale_llm_adapter import HaleLLMAdapter
from hale_llm.deploy.registry import LLM_ADAPTERS


@LLM_ADAPTERS.register_module()
class LoopedLMAdapter(HaleLLMAdapter):
    """``input_ids [B, T] → logits [B, T, V]`` with the loop unrolled ``n_loops`` times.

    Exporters trace Python loops into straight-line graphs, so the deployed depth is the
    config's ``model.loop.n_loops`` (override it via ``overrides`` to trade compute for
    quality). ACT halting with a data-dependent early exit cannot be traced; configure
    ``halting: none`` (or a fixed loop count) for export - the adapter rejects ACT configs
    with an explanatory error instead of producing a wrong graph.
    """

    variant = "looped"

    def build_model(self) -> nn.Module:
        """Build the looped model with a fixed loop count.

        Raises:
            hale_core.exceptions.ConfigError: for ACT halting, whose per-token early exit cannot be
            traced.
        """
        loop = self.run_config.model.loop
        if loop is not None and getattr(loop, "halting", "none") == "act":
            from hale_core.deploy.exceptions import ConfigError

            raise ConfigError(
                "ACT halting exits early per token (data-dependent control flow) and cannot be "
                "exported; override model.loop.halting: none (fixed depth) for deployment"
            )
        return super().build_model()

    def signature_inputs(self) -> list[dict[str, Any]]:
        """One ``input_ids [batch, seq]`` int64 input."""
        return [self.token_spec("input_ids")]

    def forward_for_export(
        self, model: nn.Module, inputs: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Forward with the loop unrolled ``n_loops`` times: ``{'logits': [B, T, V]}``."""
        return {"logits": model({"input_ids": inputs["input_ids"]})}
