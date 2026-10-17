"""Colours, tile kinds and screen regions shared by every explainer film (stage: 100 × 56.25)."""

from __future__ import annotations

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


__all__ = [
    "BG",
    "BLUE",
    "COUNTER_XY",
    "CYAN",
    "FAINT",
    "GOLD",
    "GREEN",
    "H",
    "INK",
    "KINDS",
    "MODEL_BOX",
    "MUTED",
    "ORANGE",
    "PANEL",
    "PURPLE",
    "RED",
    "ROW_GAP",
    "SEQ_TOP",
    "SPACE_GLYPH",
    "TEAL",
    "W",
    "glyph",
]
