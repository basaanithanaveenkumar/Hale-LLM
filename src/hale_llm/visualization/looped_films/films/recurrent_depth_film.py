"""Recurrent depth: train with a random loop count, then choose how long to think at test time."""

from __future__ import annotations

import random

from hale_llm.visualization.explainers.canvas import (
    CYAN,
    FAINT,
    GOLD,
    INK,
    MUTED,
    PANEL,
    PURPLE,
    RED,
    TEAL,
    GifWriter,
    ease,
)
from hale_llm.visualization.looped_films.loop_film import LoopFilm, register_loop_film
from hale_llm.visualization.looped_films.loop_lab import LoopLab

BUDGETS = (1, 2, 3, 4, 6, 8, 12, 16, 24)


class RecurrentDepthFilm(LoopFilm):
    """Prelude → shared core × T → coda; ``T`` is re-drawn every training step.

    Part 1 shows training: each step rolls a new ``T ~ U{2..8}`` and only the last 4 passes keep
    gradients (truncated backprop). Part 2 plots held-out bits per character of two character
    LMs trained on the library's docstrings, at inference budgets 1..24: the preset (no step
    embedding) stays flat far beyond the trained range; the same model with a sinusoidal step
    embedding degrades as soon as it meets steps it never saw.
    """

    name = "recurrent_depth"
    title = "Recurrent depth: thinking longer at test time"
    subtitle = "train with a random loop count T ~ U{2..8} · at inference pick T freely"

    def render(self, lab: LoopLab, writer: GifWriter) -> None:
        """Draw the film into ``writer`` from a trained ``recurrent_depth`` toy model."""
        base = {"arch": "looped", "n_layers": 1}
        run = lab.train_text({**base, "loop": {"preset": "recurrent_depth"}})
        stepped = lab.train_text(
            {**base, "loop": {"preset": "recurrent_depth", "depth_embedding": "sinusoidal"}}
        )
        losses = [lab.text_loss(run, b) for b in BUDGETS]
        other = [lab.text_loss(stepped, b) for b in BUDGETS]
        rng = random.Random(3)
        for step in range(6):  # ---- training: a new T every step
            t = rng.randint(2, 8)
            for g in range(6):
                u = ease((g + 1) / 6)
                s = self.stage(
                    "training", f"training step {step + 1}: this batch loops T = {t} times", PURPLE
                )
                self._pipeline(s, t, u, bptt=4)
                writer.stage(s, 85)
            writer.hold(250)
        for k in range(len(BUDGETS)):  # ---- inference: measured loss per budget
            for g in range(4):
                u = ease((g + 1) / 4)
                b = BUDGETS[k]
                inside = 2 <= b <= 8
                caption = f"inference with T = {b}: {losses[k]:.2f} bits/char" + (
                    ""
                    if inside
                    else (
                        "  (fewer passes than ever trained)"
                        if b < 2
                        else "  (longer than ever trained)"
                    )
                )
                s = self.stage("test-time compute", caption, GOLD)
                self._pipeline(s, b, 1.0, bptt=None)
                self._curve(s, losses[: k + 1], other[: k + 1], u)
                writer.stage(s, 85)
            writer.hold(400)
        writer.hold(2600)

    @staticmethod
    def _pipeline(s, t: int, u: float, bptt: int | None) -> None:
        y = 31.0
        s.model_box("prelude", "1 layer, once", box=(3.0, y - 4.5, 17.0, y + 4.5), color=TEAL)
        s.model_box("coda", "1 layer, once", box=(83.0, y - 4.5, 97.0, y + 4.5), color=TEAL)
        s.arrow((17.4, y), (21.0, y), color=INK, lw=1.8)
        s.arrow((79.0, y), (82.6, y), color=INK, lw=1.8)
        width = 58.0 / 16
        for k in range(16):
            x0 = 21.0 + k * width
            used = k < t
            lit = used and k <= u * t
            grad = bptt is not None and used and k >= t - bptt
            s.box(
                x0 + 0.3,
                y - 3.0,
                x0 + width - 0.3,
                y + 3.0,
                face=PANEL if not lit else "#3A2A08",
                edge=GOLD if lit else FAINT,
                lw=1.2,
                radius=0.4,
                alpha=1.0 if used else 0.35,
            )
            s.text(
                x0 + width / 2,
                y,
                "W",
                size=8,
                color=GOLD if lit else MUTED,
                weight="bold",
                alpha=1.0 if used else 0.35,
            )
            if grad and lit:
                s.text(x0 + width / 2, y - 4.2, "∇", size=9, color=CYAN, weight="bold")
        label = f"T = {t}" + (f"  ·  ∇ = gradient only through the last {bptt}" if bptt else "")
        s.text(50.0, y + 5.4, label, size=10, color=GOLD, weight="bold")

    @staticmethod
    def _curve(s, losses: list[float], other: list[float], u: float) -> None:
        x0, y0, x1, y1 = 16.0, 3.5, 84.0, 22.0
        s.box(x0, y0, x1, y1, face=PANEL, edge=FAINT, radius=1.0)
        s.text(
            x0 + 1.4,
            y1 - 1.5,
            "held-out bits per character vs passes at inference",
            size=8.5,
            color=MUTED,
            ha="left",
        )
        px0, px1, py0, py1 = x0 + 4.0, x1 - 3.0, y0 + 2.6, y1 - 3.6
        n = len(BUDGETS)
        xs = [px0 + (px1 - px0) * k / (n - 1) for k in range(n)]
        lo, hi = 2.55, 3.3
        trained = [xs[k] for k, b in enumerate(BUDGETS) if 2 <= b <= 8]
        s.box(
            trained[0] - 1.0, py0, trained[-1] + 1.0, py1, face="#1D2A1A", edge="none", radius=0.4
        )
        s.text(
            (trained[0] + trained[-1]) / 2,
            py1 - 0.8,
            "trained range T = 2..8",
            size=7.5,
            color=MUTED,
        )
        for series, color, label in (
            (other, RED, "with a step embedding"),
            (losses, GOLD, "no step embedding"),
        ):
            ys = [py0 + (py1 - py0) * (min(max(v, lo), hi) - lo) / (hi - lo) for v in series]
            if len(ys) > 1:
                s.polyline(xs[: len(ys)], ys, color=color, lw=2.0)
            for k, yy in enumerate(ys):
                s.dot(xs[k], yy, r=0.42 if k < len(ys) - 1 else 0.42 + 0.3 * (1 - u), color=color)
            if ys:
                s.text(
                    xs[len(ys) - 1] + 1.0,
                    ys[-1] + (1.0 if color == RED else -1.0),
                    label,
                    size=7.5,
                    color=color,
                    ha="left",
                )
        for k, b in enumerate(BUDGETS):
            s.text(xs[k], y0 + 1.1, str(b), size=7.5, color=MUTED)


register_loop_film(RecurrentDepthFilm)
