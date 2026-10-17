"""``hale-llm-sample``: generate one completion from a trained checkpoint."""

from __future__ import annotations

import argparse

from loguru import logger

import hale_llm.models  # noqa: F401
from hale_llm.config.experiment import apply_experiment_layout
from hale_llm.config.schema import load_config
from hale_llm.inference import load_session
from hale_llm.utils.logging import setup_logging


def main() -> None:
    """Parse arguments, load the session and print the generated text to stdout.

    ``--num-loops`` overrides ``sample.num_loops`` for looped backbones (test-time compute).
    """
    p = argparse.ArgumentParser(prog="hale-llm-sample")
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", default=None)
    p.add_argument(
        "--experiment", default=None, help="Run folder under data/experiments/<variant>/"
    )
    p.add_argument("--prompt", default=None)
    p.add_argument(
        "--num-loops",
        type=int,
        default=None,
        help="Looped backbones: passes per forward (overrides sample.num_loops)",
    )
    p.add_argument("--log-level", default=None, choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = p.parse_args()
    try:
        cfg = load_config(args.config)
        if args.log_level:
            cfg.logging.level = args.log_level
        apply_experiment_layout(cfg, create=False, name=args.experiment)
        setup_logging(level=cfg.logging.level, log_file=cfg.logging.log_file, force=True)
        session = load_session(cfg, args.checkpoint)
        result = session.generate(args.prompt, num_loops=args.num_loops)
        print(result.text)  # stdout = the generated text; diagnostics go to the log
    except Exception:
        logger.exception("hale-llm-sample failed")
        raise


if __name__ == "__main__":
    main()
