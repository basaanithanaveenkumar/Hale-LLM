"""What a slash command hands back to the chat loop."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CommandResult:
    """Text to show the user, plus whether the loop should stop."""

    message: str = ""
    exit_requested: bool = False
