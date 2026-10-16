"""Abstract base of the online-RL animations (Template Method over the explainer canvas)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from pathlib import Path

from loguru import logger

from hale_llm.visualization.explainers.canvas import GifWriter, Stage


class RLFilm(ABC):
    """Subclasses yield ``(stage, duration_ms)`` frames; :meth:`render` writes the GIF.

    Frames are drawn on the explainers' :class:`Stage` (same palette, header and typography as the
    model explainer GIFs), so the RL films read as part of the same series.
    """

    @abstractmethod
    def frames(self) -> Iterator[tuple[Stage, int]]:
        """Every frame and its display time in milliseconds."""

    def render(self, path: str | Path) -> Path:
        """Draw all frames and save a looping GIF at ``path``."""
        writer = GifWriter()
        for stage, ms in self.frames():
            writer.stage(stage, ms)
        out = writer.save(path)
        logger.info("wrote {} ({} frames, {:.1f}s)", out, len(writer), writer.seconds)
        return out
