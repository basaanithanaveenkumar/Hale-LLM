"""Deployment adapter of ``variant: diffusion`` (masked diffusion, MDLM)."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from hale_llm.deploy.adapters.hale_llm_adapter import HaleLLMAdapter
from hale_llm.deploy.registry import LLM_ADAPTERS


@LLM_ADAPTERS.register_module()
class MaskedDiffusionLMAdapter(HaleLLMAdapter):
    """``x_t [B, T], t [B] → logits [B, T, V]`` - one denoising step.

    The sampler (unmasking schedule, temperature, step count) stays in the application and calls
    the exported denoiser ``sampling_steps`` times; ``t`` is the diffusion time in ``(0, 1]``.
    Calibration feeds real text with a random ``t``-dependent fraction of tokens masked, which
    is the input distribution the denoiser actually sees.
    """

    variant = "diffusion"
    time_name = "t"
    tokens_name = "x_t"

    def __init__(self, config: str, dynamic_seq: bool = False, **kwargs: Any) -> None:
        """``dynamic_seq`` defaults to ``False``: the samplers denoise a fixed canvas, and the LGT
        backbone used by the diffusion configs guards positions with a Python ``min(seq,
        max_length)`` that ``torch.export`` cannot prove for a symbolic length (the legacy
        TorchScript ONNX exporter can still export it dynamically)."""
        super().__init__(config, dynamic_seq=dynamic_seq, **kwargs)

    def signature_inputs(self) -> list[dict[str, Any]]:
        """Noisy tokens ``x_t [batch, seq]`` and diffusion time ``t [batch]`` in ``[1e-3, 1]``."""
        return [
            self.token_spec(self.tokens_name),
            {
                "name": self.time_name,
                "shape": ["batch"],
                "dtype": "float32",
                "value_range": [1e-3, 1.0],
            },
        ]

    def forward_for_export(
        self, model: nn.Module, inputs: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """One denoising step: ``{'logits': [B, T, V]}`` for every position."""
        return {"logits": model({"x_t": inputs[self.tokens_name], "t": inputs[self.time_name]})}

    def calibration_inputs(self, token_ids: torch.Tensor) -> dict[str, torch.Tensor]:
        """Real text with a random ``t`` per row and that fraction of tokens masked."""
        mask_id = self.vocab_size - 1  # setup_tokenizer appends [MASK] as the last token
        t = torch.rand(token_ids.size(0)).clamp(0.05, 0.95)
        masked = torch.rand(token_ids.shape) < t[:, None]
        return {
            self.tokens_name: torch.where(masked, torch.full_like(token_ids, mask_id), token_ids),
            self.time_name: t,
        }
