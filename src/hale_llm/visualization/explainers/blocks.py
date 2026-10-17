"""Shared training scene of the block explainers (block diffusion, block flow matching).

Both train on ``[x_clean ; x_t]`` under the block-causal mask, with an independent time per
block, and generate one block at a time (the generation scene is
:func:`~hale_llm.visualization.explainers.masked.render_masked_generation` with ``block_size``).
:func:`two_stream_scene` sweeps the noised blocks one by one and shows what each may read: the
clean blocks before it (cyan links) and itself - never a later block - with the matching row of
the block-causal mask lighting up below.
"""

from __future__ import annotations

from collections.abc import Sequence

from hale_llm.visualization.explainers.canvas import (
    CYAN,
    FAINT,
    INK,
    MUTED,
    ORANGE,
    PANEL,
    TEAL,
    GifWriter,
    Layout,
    Stage,
    ease,
)
from hale_llm.visualization.explainers.toy import ToyRun

#: Tile geometry of the two streams (two rows of 24 each, blocks of 8).
STREAM = {"per_row": 24, "row_gap": 4.6, "tile_w": 2.9, "tile_h": 3.7, "gap": 0.5}
CLEAN_TOP, NOISED_TOP = 37.5, 24.5
#: Block-causal mask grid (stage units).
GRID_X, GRID_Y, CELL_W, CELL_H = 62.0, 2.8, 2.6, 1.55


def _mask_grid(s: Stage, num_blocks: int, active: int, alpha: float) -> None:
    """Block-level block-causal mask over ``[clean ; noised]``; row ``active`` highlighted."""
    for r in range(num_blocks):
        y = GRID_Y + (num_blocks - 1 - r) * CELL_H
        for c in range(2 * num_blocks):
            x = GRID_X + c * CELL_W + (0.8 if c >= num_blocks else 0.0)
            allowed = c < r if c < num_blocks else c - num_blocks == r
            hot = r == active
            face = (CYAN if c < num_blocks else ORANGE) if allowed else PANEL
            s.box(
                x,
                y,
                x + CELL_W - 0.3,
                y + CELL_H - 0.3,
                face=face,
                edge=FAINT if not allowed else "none",
                radius=0.2,
                alpha=(1.0 if hot else 0.3) * alpha if allowed else alpha,
            )
    top = GRID_Y + num_blocks * CELL_H + 0.9
    s.text(GRID_X + num_blocks * CELL_W / 2, top, "clean", size=8.5, color=CYAN)
    s.text(GRID_X + 1.5 * num_blocks * CELL_W + 0.8, top, "noised", size=8.5, color=ORANGE)
    s.text(
        GRID_X - 0.8,
        GRID_Y + num_blocks * CELL_H / 2,
        "row =\nnoised\nblock",
        size=7.5,
        color=MUTED,
        ha="right",
    )


def two_stream_scene(
    explainer,
    run: ToyRun,
    writer: GifWriter,
    *,
    block_size: int,
    times: Sequence[float],
    masked: Sequence[bool],
    caption: str,
) -> None:
    """Sweep the noised blocks: each reads only the clean blocks before it, and itself."""
    text = run.text
    n = len(text)
    num_blocks = n // block_size
    clean = Layout(n, block=block_size, top=CLEAN_TOP, **STREAM)
    noised = Layout(n, block=block_size, top=NOISED_TOP, **STREAM)
    frames = 8
    for b in range(num_blocks):
        for f in range(frames):
            a = ease((f + 1) / 4)
            s = explainer.stage("training", caption, TEAL)
            for i, ch in enumerate(text):
                blk = i // block_size
                x, y = clean.pos(i)
                s.tile(x, y, ch, "clean", alpha=1.0 if blk < b else 0.3, w=2.9, h=3.7, size=10.5)
                x, y = noised.pos(i)
                if masked[i]:
                    s.tile(
                        x, y, "", "mask", alpha=1.0 if blk == b else 0.45, w=2.9, h=3.7, size=10.5
                    )
                else:
                    s.tile(
                        x, y, ch, "clean", alpha=1.0 if blk == b else 0.45, w=2.9, h=3.7, size=10.5
                    )
            for side, layout in (("clean copy  x₀", clean), ("noised copy  x_t", noised)):
                s.text(
                    layout.left,
                    layout.top + 3.3,
                    side,
                    size=9.5,
                    color=CYAN if "clean" in side else ORANGE,
                    ha="left",
                )
            nx0, ny0, nx1, ny1 = noised.block_span(b, pad=0.5)
            s.box(nx0, ny0, nx1, ny1, face="none", edge=ORANGE, lw=2.2, radius=0.8, z=5)
            for c in range(b):
                cx0, cy0, cx1, cy1 = clean.block_span(c, pad=0.5)
                s.box(cx0, cy0, cx1, cy1, face="none", edge=CYAN, lw=1.6, radius=0.8, z=5, alpha=a)
                s.arc(
                    ((nx0 + nx1) / 2, ny1),
                    ((cx0 + cx1) / 2, cy0),
                    0.8,
                    color=CYAN,
                    rad=0.15,
                    alpha=a,
                )
            reads = (
                "nothing clean yet, only itself"
                if b == 0
                else (
                    f"clean block{'s' if b > 1 else ''} {', '.join(str(c) for c in range(b))}  +  itself"
                )
            )
            s.text(
                4.0,
                13.5,
                f"noised block {b}  (t = {times[b]:.2f})  reads:",
                size=11.5,
                color=ORANGE,
                ha="left",
            )
            s.text(4.0, 9.3, reads, size=12.5, weight="bold", color=INK, ha="left")
            s.text(
                4.0,
                4.8,
                "later blocks stay invisible, exactly as at generation time",
                size=10,
                color=MUTED,
                ha="left",
            )
            _mask_grid(s, num_blocks, b, 1.0)
            s.progress((b * frames + f + 1) / (num_blocks * frames), TEAL)
            writer.stage(s, 110 if b < 2 else 80)
        writer.hold(250)
    s = explainer.stage(
        "training",
        f"one forward over [clean ; noised] trains all {num_blocks} blocks at once",
        TEAL,
    )
    for i, ch in enumerate(text):
        x, y = clean.pos(i)
        s.tile(x, y, ch, "clean", w=2.9, h=3.7, size=10.5)
        x, y = noised.pos(i)
        s.tile(
            x, y, "" if masked[i] else ch, "mask" if masked[i] else "clean", w=2.9, h=3.7, size=10.5
        )
    for b in range(num_blocks):
        x0, y0, x1, y1 = noised.block_span(b, pad=0.5)
        s.text((x0 + x1) / 2, y0 - 1.2, f"t = {times[b]:.2f}", size=8.5, color=ORANGE)
    s.text(4.0, 11.0, "each block draws its own t,", size=12, color=INK, ha="left")
    s.text(4.0, 6.8, "so one batch covers every noise level", size=12, color=INK, ha="left")
    _mask_grid(s, num_blocks, -1, 1.0)
    s.progress(1.0, TEAL)
    writer.stage(s, 2600)


__all__ = ["two_stream_scene"]
