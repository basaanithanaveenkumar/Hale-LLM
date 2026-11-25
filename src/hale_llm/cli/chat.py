"""``hale-llm-chat``: talk to a trained model in the terminal.

The checkpoints are base language models, so each reply is a continuation of the
conversation so far rather than an assistant answer. Type ``/help`` for commands.

Example::

    hale-llm-chat --config configs/autoregressive/small.yaml --max-new-tokens 40
"""

from __future__ import annotations

import argparse
import sys

from loguru import logger

from hale_llm.chat import ChatCommands, ChatSession
from hale_llm.cli.session_args import add_session_arguments, open_session


def run_loop(chat: ChatSession, *, read=input, write=print) -> None:
    """Read lines until ``/quit`` or end of input. ``read``/``write`` are injectable for tests."""
    commands = ChatCommands(chat)
    write("hale-llm chat. Type /help for commands, /quit to leave.")
    while True:
        try:
            line = read("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            write("")
            return
        if not line:
            continue
        if commands.is_command(line):
            outcome = commands.execute(line)
            write(outcome.message)
            if outcome.exit_requested:
                return
            continue
        try:
            write(f"model> {chat.send(line).reply}")
        except ValueError as err:
            write(f"error: {err}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="hale-llm-chat", description=__doc__)
    # Chat is interactive, so keep the console quiet unless asked.
    add_session_arguments(parser, default_log_level="WARNING")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    try:
        session = open_session(args)
        chat = ChatSession(
            session,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )
        run_loop(chat)
    except FileNotFoundError as err:
        logger.error("{}", err)
        sys.exit(f"hale-llm-chat: {err}")
    except Exception:
        logger.exception("hale-llm-chat failed")
        raise


if __name__ == "__main__":
    main()
