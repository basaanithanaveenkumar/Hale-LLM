"""The full trainer: Ray, DDP/FSDP, W&B, resume and GIF visualisation."""

from __future__ import annotations

from loguru import logger

from hale_llm.config.schema import RunConfig
from hale_llm.training.backends.registry import register_trainer_backend
from hale_llm.training.backends.trainer_backend import TrainerBackend
from hale_llm.training.trainer import train


@register_trainer_backend("full")
class FullTrainerBackend(TrainerBackend):
    """Delegates to :func:`hale_llm.training.trainer.train`."""

    def run(self, cfg: RunConfig, tokenizer=None):
        logger.info("training backend: full")
        return train(cfg, tokenizer)
