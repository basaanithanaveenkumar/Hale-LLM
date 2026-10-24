"""Layer-sharing schedules: in which order do the passes use two shared layers A and B?"""

from __future__ import annotations

from hale_llm.visualization.explainers.canvas import (
    CYAN,
    FAINT,
    GOLD,
    INK,
    MUTED,
    PANEL,
    PURPLE,
    GifWriter,
    ease,
)
from hale_llm.visualization.looped_films.loop_film import LoopFilm, register_loop_film
from hale_llm.visualization.looped_films.loop_lab import LoopLab

SCHEDULES = (
    ("cycle", "AB AB AB AB", "the default loop: the whole core block per pass"),
    ("cycle_rev", "AB BA AB BA", "every other pass runs the core backwards"),
    ("sequence", "AA AA BB BB", "layer A for the first half of the passes, then B"),
)


class SharingFilm(LoopFilm):
    """Three schedules of the same two shared layers light up application by application.

    Parameters are identical (two layers); only the order differs - which changes what each
    layer learns to expect as input (Takase & Kiyono, 2021). ``LoopedStack.pass_layers`` builds
    exactly these schedules.
    """

    name = "sharing_patterns"
    title = "Sharing patterns for a 2-layer core"
    subtitle = "same 2 layers (A, B), 4 passes, 8 applications - three different orders"

    def render(self, lab: LoopLab, writer: GifWriter) -> None:
        """Draw the film into ``writer`` (no model needed)."""
        total = 8
        for app in range(total):
            for g in range(4):
                u = ease((g + 1) / 4)
                s = self.stage(
                    "schedules", f"application {app + 1} of {total} (pass {app // 2 + 1})", CYAN
                )
                for row, (name, order, note) in enumerate(SCHEDULES):
                    self._row(s, row, name, order.replace(" ", ""), note, app, u)
                s.footer(
                    "sharing: cycle | cycle_rev | sequence  ·  LoopedStack.pass_layers(step, "
                    "n_loops)"
                )
                s.progress((app + u) / total, GOLD)
                writer.stage(s, 80)
        writer.hold(2800)

    @staticmethod
    def _row(s, row: int, name: str, order: str, note: str, app: int, u: float) -> None:
        y = 33.0 - row * 11.0
        s.text(4.0, y + 1.0, name, size=12, weight="bold", color=INK, ha="left")
        s.text(4.0, y - 2.4, note, size=8.5, color=MUTED, ha="left")
        for k, layer in enumerate(order):
            x0 = 34.0 + k * 7.6 + (k // 2) * 1.6
            color = CYAN if layer == "A" else PURPLE
            lit = k < app or (k == app and u > 0.3)
            current = k == app
            s.box(
                x0,
                y - 2.5,
                x0 + 6.4,
                y + 2.5,
                face=PANEL,
                edge=color if lit else FAINT,
                lw=2.4 if current else 1.2,
                radius=0.6,
                alpha=1.0 if lit else 0.5,
            )
            s.text(x0 + 3.2, y, layer, size=13, weight="bold", color=color if lit else MUTED)
            if current:
                s.dot(x0 + 3.2, y + 3.6, r=0.5, color=GOLD)
            if k % 2 == 0:
                s.text(x0 + 7.0, y - 4.0, f"pass {k // 2 + 1}", size=7.5, color=MUTED)


register_loop_film(SharingFilm)
