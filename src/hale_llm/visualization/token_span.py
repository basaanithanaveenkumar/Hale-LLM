"""One coloured run of characters in a paper-style figure row."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TokenSpan:
    """A run of characters and how to draw it (prompt, revealed, masked, or highlighted)."""

    text: str
    highlight: bool = False
    muted: bool = False
    prompt: bool = False
