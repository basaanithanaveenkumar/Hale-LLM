"""Adaptive Computation Time: each character stops looping once its halting mass is spent."""

from __future__ import annotations

from collections import Counter

from hale_llm.visualization.explainers.canvas import (
    CYAN,
    FAINT,
    GOLD,
    INK,
    MUTED,
    ORANGE,
    PANEL,
    GifWriter,
    ease,
)
from hale_llm.visualization.looped_films.loop_film import LoopFilm, register_loop_film
from hale_llm.visualization.looped_films.loop_lab import LoopLab
from hale_llm.visualization.looped_films.sentence_panel import sentence_layout


class ACTFilm(LoopFilm):
    """Halting rings fill pass by pass on a trained Universal Transformer + ACT toy.

    Each character outputs ``p_t = sigmoid(w · h_t + b)`` after every pass; the ring under it
    shows the running sum. When the sum would pass ``1 - ε`` the character halts (orange), keeps
    its output and is only read by the others from then on. The ponder cost ``τ · mean(N + R)``
    pushes the model to halt early where it can.
    """

    name = "act_halting"
    title = "Adaptive Computation Time (ACT)"
    subtitle = (
        "each token decides when to stop · ring = cumulative halting probability, full ring = halt"
    )

    def render(self, lab: LoopLab, writer: GifWriter) -> None:
        """Draw the film into ``writer`` from a trained ``universal_transformer_act`` toy model."""
        run = lab.train("universal_transformer_act")
        trace = lab.trace(run)
        layout = sentence_layout(len(trace.text))
        n = len(trace.text)
        halted_at = trace.depth.tolist()
        for t in range(trace.n_passes):
            for g in range(6):
                u = ease((g + 1) / 6)
                still = int(trace.running[t].sum())
                caption = f"pass {t + 1}: {still}/{n} characters still thinking"
                s = self.stage(f"pass {t + 1}", caption, CYAN)
                for i, ch in enumerate(trace.text):
                    x, y = layout.pos(i)
                    before = trace.halting[t - 1][i] if t else 0.0
                    mass = float(before + (trace.halting[t][i] - before) * u)
                    done = halted_at[i] <= t
                    s.tile(
                        x,
                        y,
                        ch,
                        "done" if done else "clean",
                        alpha=0.55 if done else 1.0,
                        edge=ORANGE if done else (GOLD if halted_at[i] > t else None),
                    )
                    s.ring(
                        x, y - 3.9, 1.0, min(mass, 1.0), color=ORANGE if done else GOLD, width=0.45
                    )
                    if done:
                        s.text(x, y - 3.9, str(halted_at[i]), size=7, color=INK, weight="bold")
                s.model_box(
                    "shared block W",
                    f"pass {t + 1}",
                    pulse=1 - u,
                    color=GOLD,
                    box=(3.0, 4.5, 23.0, 15.5),
                )
                s.counter(f"{still}", label="still looping", color=GOLD, xy=(88.0, 11.0))
                s.footer(
                    "orange = halted (number = passes it used) · halted tokens keep being read "
                    "by the others"
                )
                s.progress((t + u) / trace.n_passes, GOLD)
                writer.stage(s, 85)
            writer.hold(350)
        self._summary(trace, layout, halted_at, writer)

    def _summary(self, trace, layout, halted_at: list[int], writer: GifWriter) -> None:
        counts = Counter(halted_at)
        mean = sum(halted_at) / len(halted_at)
        s = self.stage(
            "result",
            f"{mean:.1f} passes per character on average - not the full {max(counts)}",
            ORANGE,
        )
        for i, ch in enumerate(trace.text):
            x, y = layout.pos(i)
            s.tile(x, y, ch, "done")
            s.ring(x, y - 3.9, 1.0, 1.0, color=ORANGE, width=0.45)
            s.text(x, y - 3.9, str(halted_at[i]), size=7, color=INK, weight="bold")
        x0, y0, x1, y1 = 52.0, 5.0, 97.0, 17.5
        s.box(x0, y0, x1, y1, face=PANEL, edge=FAINT, radius=1.0)
        s.text(x0 + 1.4, y1 - 1.5, "characters by passes used", size=8.5, color=MUTED, ha="left")
        top = max(counts.values())
        keys = range(1, max(counts) + 1)
        for k in keys:
            bx = x0 + 3.0 + (k - 1) * (x1 - x0 - 6.0) / len(keys)
            h = (y1 - y0 - 5.0) * counts.get(k, 0) / top
            s.box(bx, y0 + 1.8, bx + 3.0, y0 + 1.8 + h, face=ORANGE, edge="none", radius=0.3)
            s.text(bx + 1.5, y0 + 0.9, str(k), size=7.5, color=MUTED)
        s.footer(
            "ponder cost τ·mean(N + R) is added to the loss: extra passes must pay for themselves"
        )
        writer.stage(s, 3600)


register_loop_film(ACTFilm)
