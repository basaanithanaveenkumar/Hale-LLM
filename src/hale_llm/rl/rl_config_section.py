"""Common base of the RL config sections."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class RLConfigSection(BaseModel):
    """Base of every RL config section: unknown keys are rejected (typos fail loudly)."""

    model_config = ConfigDict(extra="forbid")
