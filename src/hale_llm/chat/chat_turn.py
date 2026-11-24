"""One exchange in a chat: what the user typed and what the model continued with."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ChatTurn:
    """Immutable record of a single user message and the model's reply."""

    user: str
    reply: str
