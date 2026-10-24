"""Fixed point (DEQ view): iterate h = f(h, x) until it stops moving - if the map lets you."""

from __future__ import annotations

import torch

from hale_llm.visualization.explainers.canvas import (
    CYAN,
    FAINT,
    GOLD,
    INK,
    MUTED,
    PANEL,
    RED,
    GifWriter,
    ease,
)
from hale_llm.visualization.looped_films.loop_film import LoopFilm, register_loop_film
from hale_llm.visualization.looped_films.loop_lab import LoopLab

TOKEN = 10  # the character whose state trajectory is drawn ('m' of "model")


class FixedPointFilm(LoopFilm):
    """PCA trajectory of one token's state spiralling into ``h*``, next to the residual curves.

    Both toys use the ``deq_fixed_point`` recipe; one keeps the post-norms that make the map
    contractive, the other drops them. With them the per-pass change shrinks geometrically and the
    loop exits at the tolerance; without them it plateaus - no fixed point is reached.
    """

    name = "fixed_point"
    title = "Fixed-point view (DEQ): loop until h stops changing"
    subtitle = "h_t = f(h_{t-1}, x) converges to h* = f(h*, x) only if f is contractive"

    def render(self, lab: LoopLab, writer: GifWriter) -> None:
        """Draw the film into ``writer`` from two trained ``deq_fixed_point`` toys."""
        good_run = lab.train("deq_fixed_point", exit_tolerance=None)
        bad_run = lab.train("deq_fixed_point", exit_tolerance=None, sandwich_norm=False)
        good, bad = lab.trace(good_run), lab.trace(bad_run)
        tol = 1e-2
        res_good, res_bad = good.residuals()[1:], bad.residuals()[1:]
        exit_at = next((k for k, r in enumerate(res_good) if r < tol), len(res_good) - 1)
        points = self._pca([s[TOKEN] for s in good.states])
        n = len(res_good)
        for k in range(n):
            for g in range(4):
                u = ease((g + 1) / 4)
                done = k >= exit_at
                caption = (
                    f"pass {k + 2}: change {res_good[k]:.3f} < {tol} - fixed point reached, exit"
                    if done
                    else f"pass {k + 2}: the state still moves by {res_good[k]:.0%} of its size"
                )
                s = self.stage(f"pass {k + 2}", caption, GOLD if done else CYAN)
                self._trajectory(s, points[: k + 3], u)
                self._residuals(s, res_good[: k + 1], res_bad[: k + 1], n, tol, exit_at)
                s.footer(
                    "left: one token's hidden state, projected to 2-D · right: largest per-token "
                    "change per pass"
                )
                s.progress((k + u) / n, GOLD)
                writer.stage(s, 80)
            if k == exit_at:
                writer.hold(1500)
        writer.hold(2600)

    @staticmethod
    def _pca(states: list[torch.Tensor]) -> list[tuple[float, float]]:
        x = torch.stack(states).float()
        x = x - x.mean(0, keepdim=True)
        _, _, v = torch.pca_lowrank(x, q=2, center=False)
        p = x @ v[:, :2]
        p = (p - p.min(0).values) / (p.max(0).values - p.min(0).values).clamp_min(1e-6)
        return [(float(a), float(b)) for a, b in p]

    @staticmethod
    def _trajectory(s, pts: list[tuple[float, float]], u: float) -> None:
        x0, y0, x1, y1 = 3.0, 4.5, 45.0, 41.0
        s.box(x0, y0, x1, y1, face=PANEL, edge=FAINT, radius=1.0)
        s.text(
            x0 + 1.4,
            y1 - 1.6,
            "state of one token, pass by pass (PCA)",
            size=8.5,
            color=MUTED,
            ha="left",
        )

        def at(p):
            return x0 + 3 + (x1 - x0 - 6) * p[0], y0 + 3 + (y1 - y0 - 8) * p[1]

        xy = [at(p) for p in pts]
        if len(xy) > 1:
            s.polyline(
                [q[0] for q in xy[:-1]] + [xy[-2][0] + (xy[-1][0] - xy[-2][0]) * u],
                [q[1] for q in xy[:-1]] + [xy[-2][1] + (xy[-1][1] - xy[-2][1]) * u],
                color=GOLD,
                lw=1.6,
                alpha=0.8,
            )
        for k, (px, py) in enumerate(xy[:-1]):
            s.dot(
                px,
                py,
                r=0.35 + 0.15 * (k == 0),
                color=CYAN if k == 0 else GOLD,
                alpha=0.5 + 0.5 * k / len(xy),
            )
        s.text(xy[0][0] + 1.2, xy[0][1] + 1.2, "pass 1", size=8, color=CYAN, ha="left")

    @staticmethod
    def _residuals(
        s, good: list[float], bad: list[float], n: int, tol: float, exit_at: int
    ) -> None:
        import math

        x0, y0, x1, y1 = 50.0, 4.5, 97.0, 41.0
        s.box(x0, y0, x1, y1, face=PANEL, edge=FAINT, radius=1.0)
        s.text(
            x0 + 1.4,
            y1 - 1.6,
            "change per pass |h_t - h_t-1| / |h_t-1|  (log)",
            size=8.5,
            color=MUTED,
            ha="left",
        )
        px0, px1, py0, py1 = x0 + 4.0, x1 - 2.0, y0 + 3.0, y1 - 4.0
        lo, hi = math.log10(3e-3), math.log10(3.0)

        def at(k, r):
            v = (math.log10(max(r, 3e-3)) - lo) / (hi - lo)
            return px0 + (px1 - px0) * k / max(n - 1, 1), py0 + (py1 - py0) * v

        ty = at(0, tol)[1]
        s.polyline([px0, px1], [ty, ty], color=GOLD, lw=1.0, ls="--")
        s.text(px1, ty + 0.9, f"tolerance {tol}", size=8, color=GOLD, ha="right")
        for series, color, label in (
            (bad, RED, "no post-norm: plateaus"),
            (good, CYAN, "sandwich norm: converges"),
        ):
            pts = [at(k, r) for k, r in enumerate(series)]
            if len(pts) > 1:
                s.polyline([p[0] for p in pts], [p[1] for p in pts], color=color, lw=2.0)
            if pts:
                s.dot(*pts[-1], r=0.45, color=color)
                s.text(pts[-1][0] - 0.8, pts[-1][1] + 1.3, label, size=8, color=color, ha="right")
        if len(good) > exit_at:
            s.text(
                *at(exit_at, good[exit_at]), "  exit", size=9, color=INK, weight="bold", ha="left"
            )


register_loop_film(FixedPointFilm)
