"""GRPO optimisation section of an online RL config."""

from __future__ import annotations

from pydantic import Field

from hale_llm.rl.rl_config_section import RLConfigSection


class GRPOSection(RLConfigSection):
    """Group Relative Policy Optimisation hyper-parameters."""

    lr: float = Field(5e-4, gt=0.0)
    inner_epochs: int = Field(2, ge=1)
    clip_eps: float = Field(0.2, gt=0.0, lt=1.0)
    kl_coef: float = Field(0.02, ge=0.0)
    max_grad_norm: float = Field(1.0, gt=0.0)
    normalize_std: bool = True
    prompt_mask_prob: float = Field(0.15, ge=0.0, lt=1.0)
