"""Which GPU box to rent and where the code lives on it."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict


class JarvisSettings(BaseModel):
    """Instance settings, loaded from a YAML such as ``configs/cloud/jarvislabs.yaml``."""

    model_config = ConfigDict(extra="forbid")

    gpu_type: Literal["A30", "L4", "A100", "A100-80GB", "H100", "H200", "RTX-PRO6000"] = "A100"
    num_gpus: int = 1
    region: Literal["IN1", "IN2"] | None = "IN2"  # India regions; None lets Jarvislabs choose
    storage_gb: int = 100
    template: str = "pytorch"
    spot: bool = False  # cheaper GPU that Jarvislabs may pause when demand rises (containers only)
    name_prefix: str = "hale"
    remote_dir: str = "/home/hale-llm"  # where the code is unpacked on the instance

    @classmethod
    def from_yaml(cls, path: str | Path) -> "JarvisSettings":
        """Read the settings file."""
        return cls(**(yaml.safe_load(Path(path).read_text()) or {}))
