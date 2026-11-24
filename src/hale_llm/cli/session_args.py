"""Argument parsing and session loading shared by ``hale-llm-infer`` and ``hale-llm-chat``."""

from __future__ import annotations

import argparse

import hale_llm.models  # noqa: F401  (registers every paradigm)
from hale_llm.config.experiment import apply_experiment_layout
from hale_llm.config.schema import load_config
from hale_llm.inference import InferenceSession, load_session
from hale_llm.utils.logging import setup_logging


def add_session_arguments(parser: argparse.ArgumentParser, *, default_log_level: str | None) -> None:
    """Flags that choose which model to load and how it samples."""
    parser.add_argument("--config", required=True, help="YAML config of the trained variant")
    parser.add_argument("--checkpoint", default=None, help="Explicit checkpoint path")
    parser.add_argument("--experiment", default=None, help="Run folder under data/experiments/<variant>/")
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--steps", type=int, default=None, help="Diffusion sampling steps")
    parser.add_argument(
        "--log-level",
        default=default_log_level,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )


def open_session(args: argparse.Namespace) -> InferenceSession:
    """Load the config, bind the experiment folder, set up logging, load the model."""
    cfg = load_config(args.config)
    if args.log_level:
        cfg.logging.level = args.log_level
    apply_experiment_layout(cfg, create=False, name=args.experiment)
    setup_logging(level=cfg.logging.level, log_file=cfg.logging.log_file, force=True)
    return load_session(cfg, args.checkpoint)
