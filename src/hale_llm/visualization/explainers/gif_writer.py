"""Collect explainer frames and write one optimised, shared-palette GIF."""

from __future__ import annotations

import io
from pathlib import Path

from matplotlib.figure import Figure
from PIL import Image

from hale_llm.exceptions import VisualizationError
from hale_llm.visualization.explainers.backend import plt
from hale_llm.visualization.explainers.stage import Stage


# ---------------------------------------------------------------------------- GIF writer
class GifWriter:
    """Collects rendered frames (each with its own duration) and writes an optimised GIF."""

    def __init__(self, dpi: int = 80, colors: int = 64) -> None:
        self.dpi = dpi
        self.colors = colors
        self.frames: list[Image.Image] = []
        self.durations: list[int] = []

    def add(self, fig: Figure, duration_ms: int = 80) -> None:
        """Rasterise ``fig`` as the next frame (shown ``duration_ms``) and close it."""
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=self.dpi, facecolor=fig.get_facecolor())
        plt.close(fig)
        buf.seek(0)
        self.frames.append(Image.open(buf).convert("RGB"))
        self.durations.append(int(duration_ms))

    def stage(self, stage: Stage, duration_ms: int = 80) -> None:
        """Add a :class:`Stage` frame."""
        self.add(stage.fig, duration_ms)

    def hold(self, extra_ms: int) -> None:
        """Lengthen the last frame (a pause on a key moment)."""
        if self.durations:
            self.durations[-1] += int(extra_ms)

    def save(self, path: str | Path) -> Path:
        """Write the GIF on one shared palette and return its path (raises ``VisualizationError`` if
        empty).
        """
        if not self.frames:
            raise VisualizationError("no frames to save")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        frames = self._quantized()
        frames[0].save(
            path,
            save_all=True,
            append_images=frames[1:],
            duration=self.durations,
            loop=0,
            optimize=True,
            disposal=1,
        )
        return path

    def _quantized(self) -> list[Image.Image]:
        """All frames on ONE palette, undithered.

        A shared palette keeps unchanged pixels byte-identical between frames, so the GIF encoder
        only stores what moved (per-frame palettes made every frame a full image, ~4x larger).
        The palette is fitted on a strip of evenly spaced frames so late-appearing colours count.
        """
        sample = self.frames[:: max(1, len(self.frames) // 12)]
        w, h = sample[0].size
        strip = Image.new("RGB", (w, h * len(sample)))
        for k, frame in enumerate(sample):
            strip.paste(frame, (0, k * h))
        palette = strip.quantize(colors=self.colors, method=Image.Quantize.MEDIANCUT)
        return [frame.quantize(palette=palette, dither=Image.Dither.NONE) for frame in self.frames]

    @property
    def seconds(self) -> float:
        """Total running time in seconds."""
        return sum(self.durations) / 1000

    def __len__(self) -> int:
        return len(self.frames)
