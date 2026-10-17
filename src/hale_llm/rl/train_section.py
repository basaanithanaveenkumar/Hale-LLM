"""Loop-length and recording section of an online RL config."""

from __future__ import annotations

from pydantic import Field

from hale_llm.rl.rl_config_section import RLConfigSection


class TrainSection(RLConfigSection):
    """Loop length and what is recorded for plots and animations."""

    steps: int = Field(1000, ge=1)
    eval_every: int = Field(50, ge=1)
    track_prompts: int = Field(3, ge=0)
