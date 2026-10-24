"""Mixture-of-Recursions: a router gives every token its own number of passes up front."""

from __future__ import annotations

from collections import Counter

from hale_llm.visualization.explainers.canvas import (
    BLUE,
    CYAN,
    FAINT,
    GOLD,
    GREEN,
    INK,
    MUTED,
    PANEL,
    PURPLE,
    GifWriter,
    ease,
)
from hale_llm.visualization.looped_films.loop_film import LoopFilm, register_loop_film
from hale_llm.visualization.looped_films.loop_lab import LoopLab
from hale_llm.visualization.looped_films.sentence_panel import sentence_layout

DEPTH_COLORS = (GREEN, CYAN, BLUE, PURPLE)


class MoRFilm(LoopFilm):
    """Token-choice routing on a trained Mixture-of-Recursions toy: badges, then tokens exiting.

    The router reads each token's loop input once and picks its depth ``1..T`` (a softmax over
    depths; causal because it only looks at that token). The loop then runs ``max depth`` passes,
    but at pass ``t`` only tokens with depth ``> t`` are updated - the rest have exited.
    """

    name = "mixture_of_recursions"
    title = "Mixture-of-Recursions: a router picks each token's depth"
    subtitle = (
        "badge = passes the router granted · tokens leave the loop when their passes are used up"
    )

    def render(self, lab: LoopLab, writer: GifWriter) -> None:
        """Draw the film into ``writer`` from a trained ``mixture_of_recursions`` toy model."""
        run = lab.train("mixture_of_recursions")
        trace = lab.trace(run)
        depth = trace.depth.tolist()
        layout = sentence_layout(len(trace.text))
        n, t_max = len(depth), max(depth)
        # ---- routing: badges appear left to right
        frames = 18
        for f in range(frames):
            s = self.stage(
                "routing", "the router reads each token once and grants it 1-4 passes", CYAN
            )
            shown = ease(f / (frames - 1)) * n
            for i, ch in enumerate(trace.text):
                x, y = layout.pos(i)
                s.tile(x, y, ch, "clean")
                if i < shown:
                    self._badge(s, x, y - 3.9, depth[i], alpha=min(1.0, shown - i))
            s.model_box(
                "router", "softmax over depths", pulse=0.6, color=CYAN, box=(3.0, 4.5, 23.0, 15.5)
            )
            self._legend(s)
            writer.stage(s, 80)
        writer.hold(900)
        # ---- the loop: tokens exit as their passes run out
        used = 0
        for t in range(t_max):
            active = [d > t for d in depth]
            used += sum(active)
            for g in range(6):
                u = ease((g + 1) / 6)
                caption = f"pass {t + 1}: {sum(active)}/{n} tokens still recursing"
                s = self.stage(f"pass {t + 1} / {t_max}", caption, GOLD)
                for i, ch in enumerate(trace.text):
                    x, y = layout.pos(i)
                    s.tile(
                        x,
                        y,
                        ch,
                        "clean" if active[i] else "done",
                        alpha=1.0 if active[i] else 0.45,
                        edge=GOLD if active[i] else None,
                        glow=0.5 * (1 - u) if active[i] else 0.0,
                    )
                    self._badge(s, x, y - 3.9, depth[i], alpha=1.0 if active[i] else 0.4)
                s.model_box(
                    "shared block W",
                    f"pass {t + 1}: {sum(active)} tokens",
                    pulse=1 - u,
                    color=GOLD,
                    box=(3.0, 4.5, 23.0, 15.5),
                )
                s.counter(
                    f"{used}",
                    label=f"token-passes (full loop: {n * t_max})",
                    color=GOLD,
                    xy=(86.0, 11.0),
                )
                s.progress((t + u) / t_max, GOLD)
                writer.stage(s, 85)
            writer.hold(350)
        self._summary(trace, layout, depth, used, writer)

    @staticmethod
    def _badge(s, x: float, y: float, d: int, alpha: float = 1.0) -> None:
        color = DEPTH_COLORS[(d - 1) % len(DEPTH_COLORS)]
        s.box(
            x - 1.2,
            y - 1.0,
            x + 1.2,
            y + 1.0,
            face=color,
            edge="none",
            radius=0.5,
            alpha=0.85 * alpha,
        )
        s.text(x, y, str(d), size=8, weight="bold", color="#0B1220", alpha=alpha)

    @staticmethod
    def _legend(s) -> None:
        for k, color in enumerate(DEPTH_COLORS):
            x = 60.0 + k * 9.0
            s.box(x, 9.0, x + 2.4, 11.0, face=color, edge="none", radius=0.5)
            s.text(
                x + 3.2,
                10.0,
                f"{k + 1} pass" + ("es" if k else ""),
                size=8.5,
                color=MUTED,
                ha="left",
            )

    def _summary(self, trace, layout, depth: list[int], used: int, writer: GifWriter) -> None:
        n, t_max = len(depth), max(depth)
        saved = 1 - used / (n * t_max)
        s = self.stage(
            "result", f"{used} token-passes instead of {n * t_max}: {saved:.0%} less compute", GOLD
        )
        for i, ch in enumerate(trace.text):
            x, y = layout.pos(i)
            s.tile(x, y, ch, "done")
            self._badge(s, x, y - 3.9, depth[i])
        counts = Counter(depth)
        x0, y0, x1, y1 = 52.0, 5.0, 97.0, 17.5
        s.box(x0, y0, x1, y1, face=PANEL, edge=FAINT, radius=1.0)
        s.text(x0 + 1.4, y1 - 1.5, "tokens per granted depth", size=8.5, color=MUTED, ha="left")
        top = max(counts.values())
        for k in range(1, t_max + 1):
            bx = x0 + 4.0 + (k - 1) * 9.5
            h = (y1 - y0 - 5.0) * counts.get(k, 0) / top
            s.box(
                bx,
                y0 + 1.8,
                bx + 5.0,
                y0 + 1.8 + h,
                face=DEPTH_COLORS[k - 1],
                edge="none",
                radius=0.4,
            )
            s.text(bx + 2.5, y0 + 0.9, f"{k}: {counts.get(k, 0)}", size=7.5, color=INK)
        s.model_box(
            "router + W", "balance loss spreads depths", color=GOLD, box=(3.0, 4.5, 23.0, 15.5)
        )
        s.footer(
            "dense reference implementation: compute is masked, a production kernel would skip "
            "exited tokens"
        )
        writer.stage(s, 3600)


register_loop_film(MoRFilm)
