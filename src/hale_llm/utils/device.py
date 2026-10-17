"""Device selection with graceful fallback (CUDA → MPS → CPU)."""

from __future__ import annotations

import torch
from loguru import logger


class DeviceResolver:
    """Picks the compute device: the requested one when available, else the best present."""
    def resolve(self, device: str | None = None) -> str:
        """Resolve ``device`` (``"cuda"``, ``"mps"``, ``"cpu"`` or ``None`` for automatic).

        An unavailable MPS request falls back to CPU with a warning.
        """
        if device:
            logger.debug("using requested device={}", device)
            if device == "cuda" and not torch.cuda.is_available():
                logger.warning("cuda requested but not available; falling back")
            if device == "mps" and not (
                hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
            ):
                logger.warning("mps requested but not available; falling back to cpu")
                return "cpu"
            return device
        if torch.cuda.is_available():
            logger.info("auto-selected device=cuda")
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            logger.info("auto-selected device=mps")
            return "mps"
        logger.info("auto-selected device=cpu")
        return "cpu"


_RESOLVER = DeviceResolver()


def get_device(device: str | None = None) -> str:
    """Module-level shortcut for :meth:`DeviceResolver.resolve`."""
    return _RESOLVER.resolve(device)
