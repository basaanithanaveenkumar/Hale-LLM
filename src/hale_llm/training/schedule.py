"""Map epochs/steps config onto a concrete loop length."""

from __future__ import annotations

import math
from dataclasses import dataclass

from hale_core.exceptions import ConfigError
from loguru import logger

from hale_llm.config.schema import RunConfig
from hale_llm.exceptions import TrainingError


@dataclass(frozen=True)
class TrainSchedule:
    """Epoch and step budget of a run.

    Attributes:
        n_epochs: passes over the training loader.
        total_steps: optimisation steps (drives the LR scheduler).
    """
    n_epochs: int
    total_steps: int

    @classmethod
    def from_config(cls, cfg: RunConfig, n_batches: int) -> TrainSchedule:
        """Resolve the schedule from ``train.epochs`` (wins) or ``train.steps``.

        Raises:
            TrainingError: for an empty loader.
            hale_core.exceptions.ConfigError: when neither epochs nor steps is set.
        """
        if n_batches < 1:
            raise TrainingError("empty dataloader")
        if cfg.train.epochs is not None:
            if cfg.train.steps is not None:
                logger.warning(
                    "both epochs={} and steps={} set; using epochs ({} steps)",
                    cfg.train.epochs,
                    cfg.train.steps,
                    cfg.train.epochs * n_batches,
                )
            return cls(n_epochs=cfg.train.epochs, total_steps=cfg.train.epochs * n_batches)
        if cfg.train.steps is None:
            logger.error("set train.epochs or train.steps")
            raise ConfigError("set train.epochs or train.steps")
        return cls(n_epochs=math.ceil(cfg.train.steps / n_batches), total_steps=cfg.train.steps)


def resolve_schedule(cfg: RunConfig, n_batches: int) -> tuple[int, int]:
    """``(n_epochs, total_steps)`` of :meth:`TrainSchedule.from_config`."""
    schedule = TrainSchedule.from_config(cfg, n_batches)
    return schedule.n_epochs, schedule.total_steps
