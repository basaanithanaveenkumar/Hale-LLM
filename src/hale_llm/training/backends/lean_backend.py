"""Default backend: the single-file lean trainer."""

from __future__ import annotations

from loguru import logger

from hale_llm.config.schema import RunConfig
from hale_llm.training.context_schedule import ContextSchedule
from hale_llm.training.backends.registry import register_trainer_backend
from hale_llm.training.backends.trainer_backend import TrainerBackend
from hale_llm.training.lean_trainer import train_lean


@register_trainer_backend("lean")
class LeanTrainerBackend(TrainerBackend):
    """Runs :func:`train_lean` on one device."""

    def validate(self, cfg: RunConfig) -> None:
        # The lean loop is single-process; anything distributed needs the full trainer.
        if cfg.train.ray.enabled or cfg.train.parallel_strategy != "none":
            raise ValueError(
                "train.trainer=lean runs on one device; Ray and parallel_strategy "
                f"({cfg.train.parallel_strategy!r}) need train.trainer=full"
            )
        if cfg.train.context_stages:
            ContextSchedule._require_extensible(cfg)  # fail now rather than after data loading

    def run(self, cfg: RunConfig, tokenizer=None):
        self.validate(cfg)
        logger.info("training backend: lean")
        return train_lean(cfg, tokenizer)
