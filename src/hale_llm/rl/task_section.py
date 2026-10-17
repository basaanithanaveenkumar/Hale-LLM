"""The toy environment section of an online RL config."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from hale_llm.rl.rl_config_section import RLConfigSection


class TaskSection(RLConfigSection):
    """The toy environment: ``a+b=`` prompts with a two-digit answer."""

    name: Literal["arithmetic"] = "arithmetic"
    max_operand: int = Field(9, ge=1, le=9)
    sft_fraction: float = Field(0.4, gt=0.0, le=1.0)
