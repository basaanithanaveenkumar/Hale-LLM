"""``hale-llm-infer``: generate text for one or many prompts and print the result.

Examples::

    hale-llm-infer --config configs/block_diffusion/small.yaml --prompt "The history of"
    hale-llm-infer --config configs/autoregressive/small.yaml --prompt-file prompts.txt --json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from loguru import logger

from hale_llm.cli.session_args import add_session_arguments, open_session


def collect_prompts(prompts: list[str] | None, prompt_file: str | None) -> list[str | None]:
    """Gather prompts from ``--prompt`` flags and ``--prompt-file`` (one per line).

    ``[None]`` means "use the prompt from the config".
    """
    collected: list[str | None] = list(prompts or [])
    if prompt_file:
        lines = Path(prompt_file).read_text(encoding="utf-8").splitlines()
        collected.extend(line for line in lines if line.strip())
    return collected or [None]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="hale-llm-infer", description=__doc__)
    add_session_arguments(parser, default_log_level=None)
    parser.add_argument("--prompt", action="append", help="Prompt text; repeat for several")
    parser.add_argument("--prompt-file", help="Text file with one prompt per line")
    parser.add_argument("--json", action="store_true", help="Print one JSON object per prompt")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    try:
        prompts = collect_prompts(args.prompt, args.prompt_file)
        # Load the model once and reuse it for every prompt.
        session = open_session(args)
        for prompt in prompts:
            result = session.generate(
                prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                sampling_steps=args.steps,
            )
            if args.json:
                print(json.dumps({"prompt": result.prompt, "text": result.text}))
            else:
                print(result.text)
    except FileNotFoundError as err:
        # Missing checkpoint or prompt file: a user error, not a crash.
        logger.error("{}", err)
        sys.exit(f"hale-llm-infer: {err}")
    except Exception:
        logger.exception("hale-llm-infer failed")
        raise


if __name__ == "__main__":
    main()
