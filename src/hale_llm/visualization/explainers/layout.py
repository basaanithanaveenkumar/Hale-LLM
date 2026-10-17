"""Tile positions of a token sequence on the explainer stage."""

from __future__ import annotations

from dataclasses import dataclass

from hale_llm.visualization.explainers.palette import (
    ROW_GAP,
    SEQ_TOP,
    W,
)


# ---------------------------------------------------------------------------- layout
@dataclass(frozen=True)
class Layout:
    """Tile positions of an ``n``-token sequence: ``per_row`` tiles per row, optional blocks."""

    n: int
    per_row: int = 24
    block: int | None = None
    top: float = SEQ_TOP
    row_gap: float = ROW_GAP
    tile_w: float = 3.0
    tile_h: float = 4.3
    gap: float = 0.55
    block_gap: float = 1.4

    @property
    def step(self) -> float:
        return self.tile_w + self.gap

    @property
    def rows(self) -> int:
        return -(-self.n // self.per_row)

    def row_width(self) -> float:
        blocks = (self.per_row // self.block - 1) if self.block else 0
        return self.per_row * self.step - self.gap + blocks * self.block_gap

    @property
    def left(self) -> float:
        """Left edge of the first tile."""
        return (W - self.row_width()) / 2

    def pos(self, i: int) -> tuple[float, float]:
        """Centre of tile ``i``."""
        r, c = divmod(i, self.per_row)
        x = self.left + c * self.step + ((c // self.block) * self.block_gap if self.block else 0.0)
        return x + self.tile_w / 2, self.top - r * self.row_gap

    def block_span(self, b: int, pad: float = 0.6) -> tuple[float, float, float, float]:
        """``(x0, y0, x1, y1)`` box around block ``b`` (blocks never wrap rows)."""
        first, last = b * self.block, (b + 1) * self.block - 1
        (xa, ya), (xb, _) = self.pos(first), self.pos(last)
        return (
            xa - self.tile_w / 2 - pad,
            ya - self.tile_h / 2 - pad,
            xb + self.tile_w / 2 + pad,
            ya + self.tile_h / 2 + pad,
        )

    @property
    def bottom(self) -> float:
        """Lowest tile edge."""
        return self.pos(self.n - 1)[1] - self.tile_h / 2
