"""Build the train / validation DataLoader of packed context windows."""

from __future__ import annotations

import hashlib
import json

import torch
from torch.utils.data import DataLoader

from hale_llm.dataloader.collate import get_collate
from hale_llm.dataloader.sources.huggingface import HuggingFaceTextDataset
from hale_llm.dataloader.tokenizer import build_tokenizer_for


def dataset_fingerprint(cfg, hf_split: str) -> str:
    """Everything that changes the tokenised windows of one split (not the model variant)."""
    data = cfg.data
    size = data.train_size if hf_split == data.train_split else data.val_size
    spec = {
        "dataset": data.dataset,
        "subset": data.subset,
        "split": hf_split,
        "size": size,
        "text_field": data.text_field,
        "context_window": cfg.model.max_length,
        "packing": "non_overlapping_eos_v1",  # bump when the packing rules change
        "tokenizer": [cfg.tokenizer.source, cfg.tokenizer.kind, cfg.tokenizer.path, data.tokenizer_name],
        "long_docs": [data.long_doc_fraction, data.long_doc_min_tokens],
        "add_special_tokens": data.add_special_tokens,
    }
    return hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()


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
