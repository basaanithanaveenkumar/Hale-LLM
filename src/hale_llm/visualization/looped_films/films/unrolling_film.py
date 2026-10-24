"""Stacked vs looped: six different blocks against one block run six times."""

from __future__ import annotations

from hale_llm.visualization.explainers.canvas import (
    BLUE,
    CYAN,
    FAINT,
    GOLD,
    GREEN,
    INK,
    MUTED,
    ORANGE,
    PANEL,
    PURPLE,
    RED,
    TEAL,
    GifWriter,
    bezier,
    ease,
)
from hale_llm.visualization.looped_films.loop_film import LoopFilm, register_loop_film
from hale_llm.visualization.looped_films.loop_lab import LoopLab

DEPTH = 6
LAYER_COLORS = (CYAN, PURPLE, GREEN, ORANGE, BLUE, RED)


class UnrollingFilm(LoopFilm):
    """A state dot climbs six distinct blocks (left) and laps one shared block six times (right).

    Counters under each side make the trade-off concrete: identical compute (6 block
    applications), 6× fewer parameters for the looped model.
    """

    name = "unrolling"
    title = "Stacked vs looped transformer"
    subtitle = "same compute depth  ·  one set of weights reused instead of six"

    BOX_W, BOX_H = 14.0, 4.2
    LEFT_X, RIGHT_X = 24.0, 72.0
    BOTTOM = 9.0

    def render(self, lab: LoopLab, writer: GifWriter) -> None:
        """Draw the film into ``writer`` (no model needed)."""
        per = 7
        for step in range(DEPTH):
            for f in range(per):
                u = ease((f + 1) / per)
                self._frame(writer, step, u, final=False)
        self._frame(writer, DEPTH - 1, 1.0, final=True)
        writer.hold(2600)

    def _frame(self, writer: GifWriter, step: int, u: float, final: bool) -> None:
        caption = (
            "same 6 steps of computation - 1/6 of the parameters"
            if final
            else "a token's state goes through 6 block applications on both sides"
        )
        s = self.stage("the idea", caption, GOLD if final else TEAL)
        # ---- left: stacked, six different blocks
        s.text(self.LEFT_X, 40.0, "stacked: 6 different blocks", size=11, weight="bold", color=INK)
        for k in range(DEPTH):
            y0 = self.BOTTOM + k * (self.BOX_H + 0.9)
            lit = k <= step
            color = LAYER_COLORS[k]
            s.box(
                self.LEFT_X - self.BOX_W / 2,
                y0,
                self.LEFT_X + self.BOX_W / 2,
                y0 + self.BOX_H,
                face=PANEL,
                edge=color if lit else FAINT,
                lw=2.0 if k == step else 1.2,
                radius=0.8,
            )
            s.text(
                self.LEFT_X,
                y0 + self.BOX_H / 2,
                f"W{k + 1}",
                size=12,
                weight="bold",
                color=color if lit else MUTED,
            )
        y_from = self.BOTTOM + max(step - 1, 0) * (self.BOX_H + 0.9) + self.BOX_H / 2
        y_to = self.BOTTOM + step * (self.BOX_H + 0.9) + self.BOX_H / 2
        if not final:
            s.dot(
                self.LEFT_X - self.BOX_W / 2 - 2.0, y_from + (y_to - y_from) * u, r=0.8, color=GOLD
            )
        # ---- right: looped, one block
        s.text(self.RIGHT_X, 40.0, "looped: 1 block, 6 passes", size=11, weight="bold", color=INK)
        bx0, by0 = self.RIGHT_X - self.BOX_W / 2, 20.0
        bx1, by1 = self.RIGHT_X + self.BOX_W / 2, 20.0 + 2 * self.BOX_H
        s.box(bx0, by0, bx1, by1, face=PANEL, edge=GOLD, lw=2.4, radius=1.0)
        s.text(self.RIGHT_X, (by0 + by1) / 2 + 1.0, "W", size=16, weight="bold", color=GOLD)
        s.text(
            self.RIGHT_X, (by0 + by1) / 2 - 2.2, f"pass {step + 1} / {DEPTH}", size=9.5, color=MUTED
        )
        # The loop runs underneath the block: out of its bottom-right, back in at the bottom-left.
        exit_p, enter_p = (bx1 - 2.0, by0), (bx0 + 2.0, by0)
        ctrl = (self.RIGHT_X, by0 - 13.0)
        pts = [bezier(exit_p, ctrl, enter_p, i / 40) for i in range(41)]
        s.polyline([p[0] for p in pts], [p[1] for p in pts], color=GOLD, lw=2.6, alpha=0.8)
        s.arrow(pts[-4], (enter_p[0], enter_p[1] + 0.3), color=GOLD, lw=2.6, head=12)
        s.text(self.RIGHT_X, by0 - 9.6, "output → input", size=9, color=GOLD)
        if not final:
            s.dot(*bezier(exit_p, ctrl, enter_p, u), r=0.8, color=GOLD)
        for k in range(DEPTH):  # passes done, as pips above the block
            s.dot(
                self.RIGHT_X - 6.25 + 2.5 * k, by1 + 2.2, r=0.45, color=GOLD if k <= step else FAINT
            )
        # ---- counters
        s.counter(f"{step + 1}", label="block applications", color=TEAL, xy=(self.LEFT_X + 16, 30))
        s.counter(f"{step + 1}", label="block applications", color=TEAL, xy=(self.RIGHT_X + 17, 30))
        s.counter("6×", label="parameters (W1..W6)", color=PURPLE, xy=(self.LEFT_X + 16, 14))
        s.counter("1×", label="parameters (W)", color=GOLD, xy=(self.RIGHT_X + 17, 14))
        s.footer(
            "depth = how many times a block is applied  ·  parameters = how many different "
            "blocks exist"
        )
        s.progress((step + u) / DEPTH, GOLD)
        writer.stage(s, 70 if not final else 400)


register_loop_film(UnrollingFilm)
