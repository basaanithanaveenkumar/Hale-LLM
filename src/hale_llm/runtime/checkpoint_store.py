"""Save and load training state; callers depend on this class, not on ``torch.save``."""

from __future__ import annotations

import os
from typing import Any

import torch
from loguru import logger


class CheckpointStore:
    """Persist and restore a dict of training state (weights, optimizer, step, config)."""

    def exists(self, path: str) -> bool:
        return os.path.exists(path)

    def save(self, path: str, **state: Any) -> None:
        """Write ``state`` to ``path``, creating parent folders."""
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        # Write next to the target and rename: the rename is atomic, so a spot interruption in the
        # middle of a save leaves the previous file intact instead of a truncated one.
        temp_path = f"{path}.tmp"
        torch.save(state, temp_path)
        os.replace(temp_path, path)
        logger.info("saved checkpoint {}", path)

    def load(self, path: str, map_location: str | torch.device = "cpu") -> dict[str, Any]:
        """Read the state dict saved at ``path``."""
        if not self.exists(path):
            logger.error("checkpoint not found: {}", path)
            raise FileNotFoundError(path)
        logger.info("loading checkpoint {} map_location={}", path, map_location)
        # The file is our own training state (it includes the config), not just tensors.
        return torch.load(path, map_location=map_location, weights_only=False)
