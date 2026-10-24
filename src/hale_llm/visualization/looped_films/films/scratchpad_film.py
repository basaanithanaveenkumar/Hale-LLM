"""Scratchpad memory: learned slots that ride along every pass as extra tokens."""

from __future__ import annotations

from hale_llm.visualization.explainers.canvas import (
    CYAN,
    GOLD,
    MUTED,
    PURPLE,
    GifWriter,
    ease,
)
from hale_llm.visualization.looped_films.loop_film import LoopFilm, register_loop_film
from hale_llm.visualization.looped_films.loop_lab import LoopLab
from hale_llm.visualization.looped_films.sentence_panel import sentence_layout

SLOTS = 4
WORDS = "the tiny model learned to write"


class ScratchpadFilm(LoopFilm):
    """Who may read whom when memory slots are prepended (causal attention).

    Every token reads every slot (purple arcs). The slots read only each other (gold), because
    the same slots are shared by every position - letting them read token ``j`` would leak ``j``
    to tokens before it. With bidirectional attention (diffusion) the slots also read the text.
    """

    name = "scratchpad"
    title = "Scratchpad memory: slots carried through every pass"
    subtitle = "4 learned memory tokens, prepended inside the loop and dropped from the output"

    def render(self, lab: LoopLab, writer: GifWriter) -> None:
        """Draw the film into ``writer`` (structural; matches LoopedStack's scratch masking)."""
        layout = sentence_layout(len(WORDS))
        slot_xy = [(36.0 + k * 7.0, 13.5) for k in range(SLOTS)]
        passes = 4
        for t in range(passes):
            for g in range(8):
                u = ease((g + 1) / 8)
                s = self.stage(f"pass {t + 1} / {passes}", self._caption(g), CYAN)
                for i, ch in enumerate(WORDS):
                    x, y = layout.pos(i)
                    s.tile(x, y, ch, "clean")
                for k, (x, y) in enumerate(slot_xy):
                    s.tile(
                        x,
                        y,
                        f"m{k + 1}",
                        "draft",
                        w=5.0,
                        glow=0.6 * (1 - u) if g < 4 else 0.0,
                        size=10,
                    )
                if g < 4:  # tokens read the slots
                    for i in range(0, len(WORDS), 3):
                        tx, ty = layout.pos(i)
                        for x, y in slot_xy:
                            s.arc((tx, ty - 2.2), (x, y + 2.2), 0.35 * u, color=PURPLE, rad=0.15)
                else:  # slots read only slots
                    for a, (xa, ya) in enumerate(slot_xy):
                        for b, (xb, yb) in enumerate(slot_xy):
                            if a != b:
                                s.arc((xa, ya + 2.2), (xb, yb + 2.2), 0.6 * u, color=GOLD, rad=-0.6)
                s.text(
                    50.0,
                    8.5,
                    "slots: memory that persists across passes, invisible in the output",
                    size=9.5,
                    color=MUTED,
                )
                s.footer(
                    "causal mask: tokens → slots ✓ · slots → slots ✓ · slots → tokens ✗ (would "
                    "leak the future)"
                )
                s.progress((t + (g + 1) / 8) / passes, GOLD)
                writer.stage(s, 90)
        writer.hold(2600)

    @staticmethod
    def _caption(g: int) -> str:
        return "every token reads the memory slots" if g < 4 else "the slots update from each other"


register_loop_film(ScratchpadFilm)
