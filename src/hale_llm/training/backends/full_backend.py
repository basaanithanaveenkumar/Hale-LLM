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

    def validate(self, cfg: RunConfig) -> None:
        if cfg.train.context_stages:
            raise ValueError("train.context_stages is only supported by train.trainer=lean")

    def run(self, cfg: RunConfig, tokenizer=None):
        self.validate(cfg)
        logger.info("training backend: full")
        if cfg.train.ray.enabled and tokenizer is None:
            # Train the tokenizer once here; every Ray worker then loads it from the run folder
            # instead of all of them training (and racing to save) their own copy.
            from hale_llm.dataloader.tokenizer import build_tokenizer_for

            build_tokenizer_for(cfg)
        return train(cfg, tokenizer)
