"""Slash commands typed inside the chat loop (``/help``, ``/reset``, ...)."""

from __future__ import annotations

from collections.abc import Callable

from hale_llm.chat.chat_session import ChatSession
from hale_llm.chat.command_result import CommandResult

_HELP = """\
/help             show this list
/reset            forget the conversation
/history          print every turn so far
/temp VALUE       set sampling temperature (> 0)
/tokens N         set max new tokens per reply
/steps N          set diffusion sampling steps
/quit             leave (also /exit, Ctrl-D)"""


class ChatCommands:
    """Parses one ``/command arg`` line and applies it to a :class:`ChatSession`."""

    def __init__(self, chat: ChatSession) -> None:
        self._chat = chat
        # Dispatch table: command name -> handler taking the argument string.
        self._handlers: dict[str, Callable[[str], CommandResult]] = {
            "help": lambda _: CommandResult(_HELP),
            "reset": self._reset,
            "history": lambda _: CommandResult(self._history()),
            "temp": self._set_temperature,
            "tokens": self._set_tokens,
            "steps": self._set_steps,
            "quit": lambda _: CommandResult("bye", exit_requested=True),
            "exit": lambda _: CommandResult("bye", exit_requested=True),
        }

    @staticmethod
    def is_command(line: str) -> bool:
        return line.lstrip().startswith("/")

    def execute(self, line: str) -> CommandResult:
        """Run a command line; bad input becomes a message, never an exception."""
        name, _, argument = line.strip().lstrip("/").partition(" ")
        handler = self._handlers.get(name.lower())
        if handler is None:
            return CommandResult(f"unknown command /{name}; try /help")
        try:
            return handler(argument.strip())
        except ValueError as err:  # bad number, or the session rejected the value
            return CommandResult(f"invalid argument: {err}")

    # ----- handlers -----

    def _reset(self, _: str) -> CommandResult:
        self._chat.reset()
        return CommandResult("conversation cleared")

    def _history(self) -> str:
        if not self._chat.turns:
            return "(no turns yet)"
        return "\n".join(f"you> {t.user}\nmodel> {t.reply}" for t in self._chat.turns)

    def _set_temperature(self, argument: str) -> CommandResult:
        self._chat.temperature = float(argument)
        return CommandResult(f"temperature = {self._chat.temperature}")

    def _set_tokens(self, argument: str) -> CommandResult:
        self._chat.max_new_tokens = int(argument)
        return CommandResult(f"max new tokens = {self._chat.max_new_tokens}")

    def _set_steps(self, argument: str) -> CommandResult:
        self._chat.sampling_steps = int(argument)
        return CommandResult(f"sampling steps = {self._chat.sampling_steps}")
