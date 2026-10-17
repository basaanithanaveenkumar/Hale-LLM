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

The kit is split by responsibility: :mod:`.palette` (colours, tile kinds, regions),
:mod:`.motion` (easing and paths), :class:`~.layout.Layout`, :class:`~.stage.Stage` and
:class:`~.gif_writer.GifWriter`. This module re-exports all of them, so explainers and films
import from one place.
"""

from hale_llm.visualization.explainers.gif_writer import GifWriter
from hale_llm.visualization.explainers.layout import Layout
from hale_llm.visualization.explainers.motion import (
    arc_path,
    bezier,
    ease,
    ease_out_back,
    lerp,
    lerp2,
    pop,
)
from hale_llm.visualization.explainers.palette import (
    BG,
    BLUE,
    COUNTER_XY,
    CYAN,
    FAINT,
    GOLD,
    GREEN,
    INK,
    KINDS,
    MODEL_BOX,
    MUTED,
    ORANGE,
    PANEL,
    PURPLE,
    RED,
    ROW_GAP,
    SEQ_TOP,
    SPACE_GLYPH,
    TEAL,
    H,
    W,
    glyph,
)
from hale_llm.visualization.explainers.stage import Stage

__all__ = [
    "BG",
    "BLUE",
    "COUNTER_XY",
    "CYAN",
    "FAINT",
    "GOLD",
    "GREEN",
    "GifWriter",
    "H",
    "INK",
    "KINDS",
    "Layout",
    "MODEL_BOX",
    "MUTED",
    "ORANGE",
    "PANEL",
    "PURPLE",
    "RED",
    "ROW_GAP",
    "SEQ_TOP",
    "SPACE_GLYPH",
    "Stage",
    "TEAL",
    "W",
    "arc_path",
    "bezier",
    "ease",
    "ease_out_back",
    "glyph",
    "lerp",
    "lerp2",
    "pop",
]
