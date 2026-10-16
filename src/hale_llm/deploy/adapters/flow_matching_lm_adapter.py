"""Deployment adapter of ``variant: flow_matching`` (discrete flow matching on a masking path)."""

from __future__ import annotations

import torch

from hale_llm.deploy.adapters.masked_diffusion_lm_adapter import MaskedDiffusionLMAdapter
from hale_llm.deploy.registry import LLM_ADAPTERS


@LLM_ADAPTERS.register_module()
class FlowMatchingLMAdapter(MaskedDiffusionLMAdapter):
    """``x_t [B, T], t [B] → logits [B, T, V]`` - the posterior ``p(x_1 | x_t, t)`` of one Euler step.

    Same tensor contract as masked diffusion, but time runs ``0 → 1`` (``t = 0`` is all masks)
    and a token is real with probability ``κ(t)``; calibration samples that path.
    """

    variant = "flow_matching"

    def calibration_inputs(self, token_ids: torch.Tensor) -> dict[str, torch.Tensor]:
        mask_id = self.vocab_size - 1
        t = torch.rand(token_ids.size(0)).clamp(0.05, 0.95)
        real = torch.rand(token_ids.shape) < t[:, None]  # linear kappa(t) = t
        return {"x_t": torch.where(real, token_ids, torch.full_like(token_ids, mask_id)), "t": t}
