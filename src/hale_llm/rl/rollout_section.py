"""Rollout (self-sampling) section of an online RL config."""

from __future__ import annotations

from pydantic import Field

from hale_llm.rl.rl_config_section import RLConfigSection


class RolloutSection(RLConfigSection):
    """How the policy samples its own training data."""

    group_size: int = Field(8, ge=2)
    prompts_per_step: int = Field(16, ge=1)
    temperature: float = Field(1.5, gt=0.0)
    sampling_steps: int = Field(2, ge=1)
