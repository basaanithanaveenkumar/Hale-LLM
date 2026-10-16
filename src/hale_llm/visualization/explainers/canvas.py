"""Animation kit shared by every explainer: a dark 16:9 stage, token tiles, motion, GIF writer.

The explainers are short films, not dashboards. Every film uses the same visual grammar so the
paradigms can be compared at a glance:

* the **sequence** sits at the top - one tile per character, coloured by state (prompt, committed,
  new, masked, draft, accepted, rejected); positions that do not exist yet are dashed slots, mask
  tokens (a real input token) are solid tiles with a ``?``;
* the **model** is one box in the lower left that pulses once per forward pass;
* what the forward produced (candidate characters with their probability, guesses, drafts) appears
  next to the model and *flies* into the sequence;
* the **forward-pass counter** in the lower right is the number to compare across paradigms.

Everything is drawn on one :class:`Stage` in "stage units" (100 × 56.25) so motion can be tweened
between any two points. Frames are rendered at 800 × 450 and written with per-frame durations, so
slow moments (a real decision) and fast stretches (confident tokens) share one GIF.
"""

from __future__ import annotations

import io
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import patheffects  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Wedge  # noqa: E402
from PIL import Image  # noqa: E402

# ---------------------------------------------------------------------------- palette (dark)
BG = "#0B1220"
PANEL = "#121B2F"
INK = "#E6EDF7"
MUTED = "#8494B0"
FAINT = "#2A3654"
CYAN = "#22D3EE"
TEAL = "#2DD4BF"
GOLD = "#FBBF24"
ORANGE = "#FB923C"
PURPLE = "#A78BFA"
GREEN = "#34D399"
RED = "#F87171"
BLUE = "#60A5FA"

#: Tile kinds: (face, edge, text colour).
KINDS: dict[str, tuple[str, str, str]] = {
    "prompt": ("#1E3A64", "#60A5FA", "#DBEAFE"),  # given by the user
    "done": ("#123332", "#2DD4BF", "#CCFBF1"),  # generated earlier, now fixed context
    "new": ("#FBBF24", "#FDE68A", "#1F1300"),  # produced in this very step
    "slot": ("#0E1628", "#33415C", "#3B4A6B"),  # a position that does not exist yet (dashed)
    "mask": ("#1A2238", "#4B5A7A", "#7B8AA5"),  # the [MASK] token: a real input, unknown value
    "draft": ("#2E1F57", "#A78BFA", "#EDE9FE"),  # proposed, not yet verified
    "accept": ("#0F3D2C", "#34D399", "#D1FAE5"),
    "reject": ("#4A1520", "#F87171", "#FECACA"),
    "ghost": ("#0F172A", "#1E293B", "#334155"),  # out of view / not visible to this step
    "clean": ("#1B2A44", "#475569", "#CBD5E1"),  # plain training text
}

SPACE_GLYPH = "␣"
W, H = 100.0, 56.25

#: Shared screen regions (stage units): sequence rows on top, model + outputs below.
SEQ_TOP = 37.0
ROW_GAP = 6.4
MODEL_BOX = (3.0, 8.5, 22.0, 21.5)
COUNTER_XY = (90.0, 15.5)


def glyph(char: str) -> str:
    """Display form of a character token (spaces become ``␣``)."""
    return SPACE_GLYPH if char == " " else char


# ---------------------------------------------------------------------------- easing
def ease(t: float) -> float:
    """Smooth in-out (cubic), clamped to ``[0, 1]``."""
    t = min(max(t, 0.0), 1.0)
    return 4 * t**3 if t < 0.5 else 1 - (-2 * t + 2) ** 3 / 2


def ease_out_back(t: float) -> float:
    """Overshoot then settle - the 'pop' of a committed token."""
    t = min(max(t, 0.0), 1.0)
    c1, c3 = 1.70158, 2.70158
    return 1 + c3 * (t - 1) ** 3 + c1 * (t - 1) ** 2


def pop(t: float, amount: float = 0.3) -> float:
    """Scale of a tile that pops in: grows past 1 by ``amount`` and settles back."""
    return 1.0 + amount * math.sin(math.pi * min(max(t, 0.0), 1.0))


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def lerp2(p: tuple[float, float], q: tuple[float, float], t: float) -> tuple[float, float]:
    return lerp(p[0], q[0], t), lerp(p[1], q[1], t)


def arc_path(p: tuple[float, float], q: tuple[float, float], t: float, lift: float = 6.0) -> tuple[float, float]:
    """Point at ``t`` on a parabolic flight from ``p`` to ``q`` (bulging up by ``lift``)."""
    x, y = lerp2(p, q, t)
    return x, y + lift * 4 * t * (1 - t)


def bezier(p: tuple[float, float], c: tuple[float, float], q: tuple[float, float], t: float) -> tuple[float, float]:
    """Point at ``t`` on the quadratic Bézier ``p → q`` with control point ``c``."""
    a, b = (1 - t) ** 2, 2 * (1 - t) * t
    return a * p[0] + b * c[0] + t * t * q[0], a * p[1] + b * c[1] + t * t * q[1]


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
        return (xa - self.tile_w / 2 - pad, ya - self.tile_h / 2 - pad,
                xb + self.tile_w / 2 + pad, ya + self.tile_h / 2 + pad)

    @property
    def bottom(self) -> float:
        """Lowest tile edge."""
        return self.pos(self.n - 1)[1] - self.tile_h / 2


# ---------------------------------------------------------------------------- stage
class Stage:
    """One frame: header (title, subtitle, phase pill), a caption and a drawing surface."""

    def __init__(self, title: str, subtitle: str, phase: str, caption: str = "",
                 phase_color: str = TEAL) -> None:
        self.fig = plt.figure(figsize=(10, 5.625), facecolor=BG)
        self.ax = self.fig.add_axes((0, 0, 1, 1))
        self.ax.set_xlim(0, W)
        self.ax.set_ylim(0, H)
        self.ax.axis("off")
        self.text(3, 52.6, title, size=19, weight="bold", ha="left")
        self.text(3, 49.0, subtitle, size=10.5, color=MUTED, ha="left")
        pill = self.ax.text(97, 52.6, f" {phase} ", fontsize=11, weight="bold", color=BG, ha="right",
                            va="center")
        pill.set_bbox({"boxstyle": "round,pad=0.38", "facecolor": phase_color, "edgecolor": "none"})
        if caption:
            self.text(50, 43.4, caption, size=14, weight="bold", color=INK)

    # ------------------------------------------------------------------ primitives
    def text(self, x: float, y: float, s: str, size: float = 11, color: str = INK, ha: str = "center",
             va: str = "center", weight: str = "normal", alpha: float = 1.0, family: str = "DejaVu Sans",
             **kw):
        return self.ax.text(x, y, s, fontsize=size, color=color, ha=ha, va=va, weight=weight,
                            alpha=min(max(alpha, 0.0), 1.0), family=family, **kw)

    def box(self, x0: float, y0: float, x1: float, y1: float, face: str = PANEL, edge: str = FAINT,
            lw: float = 1.0, radius: float = 0.8, alpha: float = 1.0, ls: str | tuple = "-", z: float = 1):
        self.ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0,
                                         boxstyle=f"round,pad=0,rounding_size={radius}", facecolor=face,
                                         edgecolor=edge, linewidth=lw, alpha=min(max(alpha, 0.0), 1.0),
                                         linestyle=ls, zorder=z))

    def tile(self, x: float, y: float, char: str = "", kind: str = "done", *, scale: float = 1.0,
             alpha: float = 1.0, glow: float = 0.0, w: float = 3.0, h: float = 4.3, size: float = 12.5,
             ghost: str = "", ghost_alpha: float = 0.0, label: str = "", label_color: str = MUTED,
             edge: str | None = None, z: float = 3) -> None:
        """A token tile centred at ``(x, y)``.

        ``char`` is drawn solid; otherwise ``ghost`` (a tentative guess) is drawn in italics with
        opacity ``ghost_alpha``; otherwise a mask tile shows ``?``. ``label`` goes under the tile.
        """
        face, default_edge, color = KINDS[kind]
        sw, sh = w * scale, h * scale
        if glow > 0:
            for k, a in ((1.0, 0.12), (0.5, 0.22)):
                self.box(x - sw / 2 - k, y - sh / 2 - k, x + sw / 2 + k, y + sh / 2 + k,
                         face=edge or default_edge, edge="none", radius=1.2, alpha=a * glow * alpha, z=z - 0.5)
        self.box(x - sw / 2, y - sh / 2, x + sw / 2, y + sh / 2, face=face, edge=edge or default_edge, lw=1.3,
                 radius=0.7, alpha=alpha, ls=(0, (2.5, 2)) if kind == "slot" else "-", z=z)
        mono = "DejaVu Sans Mono"
        if char:
            self.text(x, y, glyph(char), size=size * scale, color=color, family=mono, weight="bold", alpha=alpha,
                      zorder=z + 0.2)
        elif ghost and ghost_alpha > 0.02:
            self.text(x, y, glyph(ghost), size=size * scale, color=GOLD, family=mono, style="italic",
                      alpha=alpha * (0.2 + 0.8 * ghost_alpha), zorder=z + 0.2)
        elif kind == "mask":
            self.text(x, y, "?", size=size * 0.85 * scale, color=color, family=mono, alpha=alpha, zorder=z + 0.2)
        if label:
            self.text(x, y - h / 2 - 1.1, label, size=7.5, color=label_color, alpha=alpha, zorder=z)

    def arc(self, p: tuple[float, float], q: tuple[float, float], weight: float, color: str = CYAN,
            rad: float = -0.3, lw: float = 2.6, alpha: float = 1.0) -> None:
        """Curved connection whose opacity and width follow ``weight`` (0..1)."""
        if weight < 0.03 or alpha <= 0.01:
            return
        self.ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-", connectionstyle=f"arc3,rad={rad}",
                                          color=color, linewidth=0.6 + lw * weight,
                                          alpha=min(1.0, (0.15 + 0.8 * weight) * alpha), zorder=2))

    def arrow(self, p: tuple[float, float], q: tuple[float, float], color: str = MUTED, lw: float = 1.6,
              rad: float = 0.0, alpha: float = 1.0, head: float = 10) -> None:
        if alpha <= 0.01:
            return
        self.ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=head,
                                          connectionstyle=f"arc3,rad={rad}", color=color, linewidth=lw,
                                          alpha=min(max(alpha, 0.0), 1.0), zorder=2))

    def bar(self, x: float, y: float, w: float, h: float, frac: float, color: str = TEAL,
            back: str = FAINT, alpha: float = 1.0) -> None:
        self.box(x, y, x + w, y + h, face=back, edge="none", radius=min(h / 2, 0.5), alpha=alpha)
        if frac > 0.003:
            self.box(x, y, x + w * min(frac, 1.0), y + h, face=color, edge="none", radius=min(h / 2, 0.5),
                     alpha=alpha)

    def ring(self, cx: float, cy: float, r: float, frac: float, color: str = GOLD, width: float = 0.6) -> None:
        """A ring filled clockwise to ``frac``."""
        self.ax.add_patch(Circle((cx, cy), r, facecolor="none", edgecolor=FAINT, lw=2.0, zorder=3))
        if frac > 0.002:
            self.ax.add_patch(Wedge((cx, cy), r + width / 2, 90 - 360 * min(frac, 1.0), 90, width=width,
                                    facecolor=color, edgecolor="none", zorder=4))

    def mark(self, x: float, y: float, ok: bool, scale: float = 1.0, alpha: float = 1.0) -> None:
        t = self.text(x, y, "✓" if ok else "✗", size=13 * scale, weight="bold", color=GREEN if ok else RED,
                      alpha=alpha, zorder=6)
        t.set_path_effects([patheffects.withStroke(linewidth=3, foreground=BG)])

    def dot(self, x: float, y: float, r: float = 0.6, color: str = GOLD, alpha: float = 1.0) -> None:
        self.ax.add_patch(Circle((x, y), r * 2.4, facecolor=color, edgecolor="none", alpha=0.18 * alpha, zorder=5))
        self.ax.add_patch(Circle((x, y), r, facecolor=color, edgecolor="none", alpha=alpha, zorder=6))

    def polyline(self, xs: Sequence[float], ys: Sequence[float], color: str = MUTED, lw: float = 2.0,
                 alpha: float = 1.0, ls: str = "-", z: float = 2) -> None:
        self.ax.plot(xs, ys, color=color, linewidth=lw, alpha=min(max(alpha, 0.0), 1.0), linestyle=ls,
                     zorder=z, solid_capstyle="round")

    def footer(self, s: str, alpha: float = 1.0) -> None:
        """A one-line legend at the bottom left."""
        self.text(3, 2.6, s, size=9.5, color=MUTED, ha="left", alpha=alpha)

    def progress(self, frac: float, color: str = TEAL) -> None:
        """Thin progress line at the very bottom of the frame."""
        self.box(0, 0, W, 0.6, face=FAINT, edge="none", radius=0)
        self.box(0, 0, W * min(max(frac, 0.0), 1.0), 0.6, face=color, edge="none", radius=0)

    # ------------------------------------------------------------------ composite widgets
    def model_box(self, name: str, detail: str = "", pulse: float = 0.0, color: str = CYAN,
                  box: tuple[float, float, float, float] = MODEL_BOX) -> None:
        """The network: a panel that glows while a forward pass runs (``pulse`` in 0..1)."""
        x0, y0, x1, y1 = box
        if pulse > 0:
            for k, a in ((1.4, 0.10), (0.7, 0.2)):
                self.box(x0 - k, y0 - k, x1 + k, y1 + k, face=color, edge="none", radius=1.6, alpha=a * pulse)
        self.box(x0, y0, x1, y1, face=PANEL, edge=color if pulse > 0.05 else FAINT, lw=1.6 + 1.2 * pulse,
                 radius=1.2, z=1.5)
        cx = (x0 + x1) / 2
        self.text(cx, (y0 + y1) / 2 + (1.4 if detail else 0), name, size=13, weight="bold",
                  color=INK if pulse < 0.05 else color, zorder=3)
        if detail:
            self.text(cx, (y0 + y1) / 2 - 2.0, detail, size=8.5, color=MUTED, zorder=3)

    def counter(self, value: str, label: str = "forward passes", color: str = GOLD,
                xy: tuple[float, float] = COUNTER_XY, sub: str = "") -> None:
        """The one number to compare across paradigms."""
        x, y = xy
        self.text(x, y + 1.6, value, size=30, weight="bold", color=color)
        self.text(x, y - 3.4, label, size=9.5, color=MUTED)
        if sub:
            self.text(x, y - 6.0, sub, size=9.5, color=INK)

    def candidates(self, x: float, y: float, items: Sequence[tuple[str, float]], *, chosen: int | None = None,
                   grow: float = 1.0, alpha: float = 1.0, hide: int | None = None, gap: float = 4.6,
                   w: float = 13.0, title: str = "next character") -> list[tuple[float, float]]:
        """Candidate tokens with probability bars, top to bottom; returns each tile's centre.

        ``grow`` animates the bars from 0, ``chosen`` highlights one row in gold, ``hide`` leaves one
        tile out (it is flying somewhere else).
        """
        if title:
            self.text(x - 1.5, y + 3.6, title, size=9, color=MUTED, ha="left", alpha=alpha)
        out = []
        for j, (char, prob) in enumerate(items):
            cy = y - j * gap
            out.append((x, cy))
            hl = chosen == j
            a = alpha * (1.0 if chosen is None or hl else 0.45)
            if hide != j:
                self.tile(x, cy, char, "new" if hl else "clean", alpha=a, glow=0.8 if hl else 0)
            bx = x + 2.4
            self.bar(bx, cy - 0.7, w, 1.4, prob * grow, color=GOLD if hl else CYAN, alpha=a)
            self.text(bx + w + 0.9, cy, f"{prob * grow:.0%}", size=10.5, color=INK if hl else MUTED, ha="left",
                      alpha=a, weight="bold" if hl else "normal")
        return out

    def schedule(self, box: tuple[float, float, float, float], fn: Callable[[float], float], t: float, *,
                 title: str, value: str, reverse: bool = False, color: str = GOLD,
                 extra: tuple[Callable[[float], float], str] | None = None, alpha: float = 1.0,
                 dots: Sequence[float] = ()) -> None:
        """A mini plot of ``fn`` on ``t ∈ [0, 1]`` with a dot at ``t``.

        ``reverse`` draws the time axis right to left (time running 1 → 0); ``extra`` adds a
        dashed comparison curve; ``dots`` marks further times (e.g. one per training block).
        """
        x0, y0, x1, y1 = box
        self.box(x0, y0, x1, y1, face=PANEL, edge=FAINT, radius=1.0, alpha=alpha)
        px0, px1, py0, py1 = x0 + 3.0, x1 - 1.6, y0 + 2.6, y1 - 3.6

        def at(tt: float, v: float) -> tuple[float, float]:
            u = 1 - tt if reverse else tt
            return lerp(px0, px1, u), lerp(py0, py1, min(max(v, 0.0), 1.0))

        grid = [i / 60 for i in range(61)]
        self.polyline([px0, px1], [py0, py0], color=FAINT, lw=1.2, alpha=alpha)
        self.polyline([px0, px0], [py0, py1], color=FAINT, lw=1.2, alpha=alpha)
        if extra is not None:
            pts = [at(g, extra[0](g)) for g in grid]
            self.polyline([p[0] for p in pts], [p[1] for p in pts], color=MUTED, lw=1.4, ls="--", alpha=alpha)
            lx, ly = at(0.45, extra[0](0.45))
            self.text(lx - 0.6, ly + 1.0, extra[1], size=8, color=MUTED, ha="right", alpha=alpha)
        pts = [at(g, fn(g)) for g in grid]
        self.polyline([p[0] for p in pts], [p[1] for p in pts], color=INK, lw=2.0, alpha=alpha)
        for d in dots:
            self.dot(*at(d, fn(d)), r=0.45, color=ORANGE, alpha=alpha)
        dx, dy = at(t, fn(t))
        self.polyline([dx, dx], [py0, dy], color=color, lw=1.0, ls=":", alpha=alpha)
        self.dot(dx, dy, r=0.6, color=color, alpha=alpha)
        self.text(x0 + 1.4, y1 - 1.7, title, size=9, color=MUTED, ha="left", alpha=alpha)
        self.text(x1 - 1.4, y1 - 1.7, value, size=10, color=color, ha="right", weight="bold", alpha=alpha)
        left, right = ("t = 1", "t = 0") if reverse else ("t = 0", "t = 1")
        self.text(px0, y0 + 1.1, left, size=7.5, color=MUTED, alpha=alpha)
        self.text(px1, y0 + 1.1, right, size=7.5, color=MUTED, alpha=alpha)


# ---------------------------------------------------------------------------- GIF writer
class GifWriter:
    """Collects rendered frames (each with its own duration) and writes an optimised GIF."""

    def __init__(self, dpi: int = 80, colors: int = 64) -> None:
        self.dpi = dpi
        self.colors = colors
        self.frames: list[Image.Image] = []
        self.durations: list[int] = []

    def add(self, fig: Figure, duration_ms: int = 80) -> None:
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=self.dpi, facecolor=fig.get_facecolor())
        plt.close(fig)
        buf.seek(0)
        self.frames.append(Image.open(buf).convert("RGB"))
        self.durations.append(int(duration_ms))

    def stage(self, stage: Stage, duration_ms: int = 80) -> None:
        self.add(stage.fig, duration_ms)

    def hold(self, extra_ms: int) -> None:
        """Lengthen the last frame (a pause on a key moment)."""
        if self.durations:
            self.durations[-1] += int(extra_ms)

    def save(self, path: str | Path) -> Path:
        if not self.frames:
            raise ValueError("no frames to save")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        frames = self._quantized()
        frames[0].save(path, save_all=True, append_images=frames[1:], duration=self.durations, loop=0,
                       optimize=True, disposal=1)
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
        return sum(self.durations) / 1000

    def __len__(self) -> int:
        return len(self.frames)


__all__ = [
    "BG", "BLUE", "COUNTER_XY", "CYAN", "FAINT", "GOLD", "GREEN", "INK", "KINDS", "MODEL_BOX", "MUTED", "ORANGE",
    "PANEL", "PURPLE", "RED", "ROW_GAP", "SEQ_TOP", "TEAL", "GifWriter", "Layout", "Stage", "arc_path", "bezier",
    "ease", "ease_out_back", "glyph", "lerp", "lerp2", "pop",
]
