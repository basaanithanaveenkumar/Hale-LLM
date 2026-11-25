"""Build the train / validation DataLoader of packed context windows."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from hale_llm.dataloader.collate import get_collate
from hale_llm.dataloader.sources.huggingface import HuggingFaceTextDataset
from hale_llm.dataloader.tokenizer import build_tokenizer_for


def make_dataloader(cfg, split: str = "train", **collate_kwargs) -> DataLoader:
    """Tokenise one split into context windows and wrap it in a batching DataLoader."""
    tokenizer = build_tokenizer_for(cfg)
    hf_split = cfg.data.train_split if split == "train" else cfg.data.val_split
    dataset = HuggingFaceTextDataset(tokenizer, cfg, split=hf_split)
    collate = get_collate(cfg.variant)

    def _collate(batch):
        tensors = torch.stack(list(batch), dim=0)
        return collate(tensors, tokenizer=tokenizer, **collate_kwargs)

    return DataLoader(
        dataset,
        batch_size=cfg.train.batch_size,
        drop_last=True,
        collate_fn=_collate,
        shuffle=split == "train",
    )
