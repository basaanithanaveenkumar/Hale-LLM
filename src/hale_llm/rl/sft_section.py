"""Supervised warm-start section of an online RL config."""

from __future__ import annotations

from pydantic import Field

from hale_llm.rl.rl_config_section import RLConfigSection


class SFTSection(RLConfigSection):
    """Supervised warm start on a fraction of the facts (gives the policy a format and a prior)."""

    steps: int = Field(60, ge=0)
    lr: float = Field(3e-3, gt=0.0)
