"""Build train/val loaders from config. Trainer depends on this, not on Hub details."""

from __future__ import annotations

from torch.utils.data import DataLoader

from hale_llm.config.schema import RunConfig
from hale_llm.dataloader.loaders import make_dataloader
from hale_llm.dataloader.sources.overfit import make_overfit_loader, uses_overfit
from hale_llm.dataloader.tokenizer import build_tokenizer_for
from loguru import logger


class DataModule:
    def __init__(self, cfg: RunConfig, tokenizer=None) -> None:
        self.cfg = cfg
        self.tokenizer = tokenizer or build_tokenizer_for(cfg)

    def train_loader(self) -> DataLoader:
        cfg = self.cfg
        if uses_overfit(cfg):
            if not cfg.data.overfit_text:
                raise ValueError("data.source=overfit requires data.overfit_text")
            logger.warning("overfit mode enabled; HuggingFace loader is bypassed")
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
        if uses_overfit(self.cfg):
            if train_loader is None:
                return self.train_loader()
            return train_loader
        return make_dataloader(self.cfg, split="validation")
