"""Thin CLI: load config → import models (registration) → train."""

from __future__ import annotations

import argparse

import hale_llm.models  # noqa: F401  — triggers registry
from loguru import logger

from hale_llm.config.experiment import apply_experiment_layout
from hale_llm.config.schema import load_config
from hale_llm.training.backends import get_trainer_backend
from hale_llm.utils.logging import setup_logging


def _warn_ignored_by_lean(args: argparse.Namespace) -> None:
    """The lean loop has no resume, W&B or GIFs; say so instead of silently ignoring flags."""
    ignored = [
        flag
        for flag, used in {
            "--resume": args.resume,
            "--viz": args.viz_enabled,
            "--wandb": args.wandb,
        }.items()
        if used
    ]
    if ignored:
        logger.warning("lean trainer ignores {}; use --trainer full for these", ", ".join(ignored))


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
        help="Backend: lean (default) | full | torchtitan. Overrides train.trainer",
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
    viz = p.add_mutually_exclusive_group()
    viz.add_argument(
        "--viz",
        dest="viz_enabled",
        action="store_true",
        default=None,
        help="Write a generation GIF every viz.every_n_steps (overrides viz.enabled)",
    )
    viz.add_argument(
        "--no-viz",
        dest="viz_enabled",
        action="store_false",
        help="Disable GIFs during training",
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
    
    wandb_group = p.add_mutually_exclusive_group()
    wandb_group.add_argument(
        "--wandb",
        dest="wandb",
        action="store_true",
        default=None,
        help="Enable Weights & Biases logging",
    )
    wandb_group.add_argument(
        "--no-wandb",
        dest="wandb",
        action="store_false",
        help="Disable Weights & Biases logging",
    )
    
    p.add_argument("--wandb-project", default=None, help="W&B project name")
    p.add_argument("--wandb-entity", default=None, help="W&B entity (username or team)")
    p.add_argument("--wandb-run-name", default=None, help="W&B run name")
    p.add_argument("--wandb-tags", nargs="+", default=None, help="W&B tags for the run")
    
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
        if args.viz_enabled is not None:
            cfg.viz.enabled = args.viz_enabled
        
        # Logging overrides
        if args.tensorboard is not None:
            cfg.logging.tensorboard = args.tensorboard
        if args.wandb is not None:
            cfg.logging.wandb = args.wandb
        if args.wandb_project is not None:
            cfg.logging.wandb_project = args.wandb_project
        if args.wandb_entity is not None:
            cfg.logging.wandb_entity = args.wandb_entity
        if args.wandb_run_name is not None:
            cfg.logging.wandb_run_name = args.wandb_run_name
        if args.wandb_tags is not None:
            cfg.logging.wandb_tags = args.wandb_tags
        
        if args.trainer is not None:
            cfg.train.trainer = args.trainer
        backend = get_trainer_backend(cfg.train.trainer)  # unknown name -> KeyError listing options
        backend.validate(cfg)  # fail before creating any run folder
        if cfg.train.trainer == "lean":
            _warn_ignored_by_lean(args)
        apply_experiment_layout(cfg, create=True, name=args.experiment)
        setup_logging(level=cfg.logging.level, log_file=cfg.logging.log_file, force=True)
        logger.info(
            "hale-llm-train config={} resume={} viz={} experiment={}",
            args.config,
            cfg.train.resume,
            cfg.viz.enabled,
            cfg.experiment.name,
        )
        backend.run(cfg)
    except Exception:
        logger.exception("hale-llm-train failed")
        raise


if __name__ == "__main__":
    main()
