"""Thin CLI: load config → import models (registration) → train."""

from __future__ import annotations

import argparse

import hale_llm.models  # noqa: F401  — triggers registry
from loguru import logger

from hale_llm.config.experiment import apply_experiment_layout
from hale_llm.config.schema import load_config
from hale_llm.training.backends import get_trainer_backend
from hale_llm.utils.logging import setup_logging


def main() -> None:
    p = argparse.ArgumentParser(prog="hale-llm-train")
    p.add_argument("--config", required=True, help="YAML under configs/")
    p.add_argument("--experiment", default=None, help="Reuse this run folder under data/experiments/<variant>/")
    p.add_argument("--log-level", default=None, choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    p.add_argument("--epochs", type=int, default=None, help="Override train.epochs")
    p.add_argument("--steps", type=int, default=None, help="Override train.steps (ignored if --epochs is set)")
    p.add_argument(
        "--trainer",
        default=None,
        help="Backend: lean (default) | torchtitan. Overrides train.trainer",
    )
    resume = p.add_mutually_exclusive_group()
    resume.add_argument(
        "--resume",
        dest="resume",
        action="store_true",
        default=None,
        help="Resume from train.checkpoint_path if it exists (default from config)",
    )
    resume.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        help="Ignore existing checkpoint and train from scratch",
    )
    # Logging arguments
    tb = p.add_mutually_exclusive_group()
    tb.add_argument(
        "--tensorboard",
        dest="tensorboard",
        action="store_true",
        default=None,
        help="Enable TensorBoard logging",
    )
    tb.add_argument(
        "--no-tensorboard",
        dest="tensorboard",
        action="store_false",
        help="Disable TensorBoard logging",
    )
    
    args = p.parse_args()
    try:
        cfg = load_config(args.config)
        if args.log_level:
            cfg.logging.level = args.log_level
        if args.epochs is not None:
            cfg.train.epochs = args.epochs
            cfg.train.steps = None
        elif args.steps is not None:
            cfg.train.steps = args.steps
            cfg.train.epochs = None
        if args.resume is not None:
            cfg.train.resume = args.resume
        # Logging overrides
        if args.tensorboard is not None:
            cfg.logging.tensorboard = args.tensorboard
        if args.trainer is not None:
            cfg.train.trainer = args.trainer
        backend = get_trainer_backend(cfg.train.trainer)  # unknown name -> KeyError listing options
        backend.validate(cfg)  # fail before creating any run folder
        apply_experiment_layout(cfg, create=True, name=args.experiment)
        setup_logging(level=cfg.logging.level, log_file=cfg.logging.log_file, force=True)
        logger.info(
            "hale-llm-train config={} variant={} experiment={}",
            args.config,
            cfg.variant,
            cfg.experiment.name,
        )
        backend.run(cfg)
    except Exception:
        logger.exception("hale-llm-train failed")
        raise


if __name__ == "__main__":
    main()
