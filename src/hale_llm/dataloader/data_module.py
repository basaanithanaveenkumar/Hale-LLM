"""Build train/val loaders from config. Trainer depends on this, not on Hub details."""

from __future__ import annotations

from loguru import logger
from torch.utils.data import DataLoader

from hale_llm.config.schema import RunConfig
from hale_llm.dataloader.loaders import make_dataloader
from hale_llm.dataloader.prompts import collect_dataset_passages, dataset_prompt
from hale_llm.dataloader.sources.overfit import make_overfit_loader, uses_overfit
from hale_llm.dataloader.tokenizer import setup_tokenizer
from hale_llm.exceptions import DataPipelineError


class DataModule:
    """Builds the train/validation loaders and visualisation prompts of one run.

    The trainer depends on this class, not on Hub or dataset details; overfit runs reuse the
    training loader for validation.

    Args:
        cfg: the run config.
        tokenizer: optional pre-loaded tokenizer (default: ``data.tokenizer_name``).
    """
    def __init__(self, cfg: RunConfig, tokenizer=None) -> None:
        self.cfg = cfg
        self.tokenizer = tokenizer or setup_tokenizer(cfg.data.tokenizer_name)
        self._viz_passages: list[str] | None = None

    def viz_prompt(self, seed: int = 0) -> str:
        """Prompt for a training-time GIF: a dataset passage prefix, or ``sample.prompt``.

        Args:
            seed: picks the passage, so successive GIFs use different prompts.
        """
        if self.cfg.viz.prompt_source != "dataset":
            return self.cfg.sample.prompt
        if self._viz_passages is None:
            self._viz_passages = collect_dataset_passages(self.cfg, min_words=self.cfg.viz.prompt_words)
        return dataset_prompt(self.cfg, seed=seed, passages=self._viz_passages)

    def train_loader(self) -> DataLoader:
        """Shuffled, last-batch-dropped training loader for the configured source.

        Raises:
            DataPipelineError: for ``source: overfit`` without ``overfit_text``.
        """
        cfg = self.cfg
        if uses_overfit(cfg):
            if not cfg.data.overfit_text:
                raise DataPipelineError("data.source=overfit requires data.overfit_text")
            logger.warning("overfit mode enabled; HuggingFace loader is bypassed")
            logger.debug("overfit_text={!r} copies={}", cfg.data.overfit_text, cfg.data.n_overfit_copies)
            return make_overfit_loader(
                cfg.variant,
                self.tokenizer,
                cfg.data.overfit_text,
                cfg.model.max_length,
                cfg.train.batch_size,
                cfg.data.n_overfit_copies,
            )
        return make_dataloader(cfg, split="train")

    def val_loader(self, train_loader: DataLoader | None = None) -> DataLoader:
        """Validation loader; in overfit mode the training loader (``train_loader`` if given)."""
        if uses_overfit(self.cfg):
            if train_loader is None:
                return self.train_loader()
            return train_loader
        return make_dataloader(self.cfg, split="validation")
