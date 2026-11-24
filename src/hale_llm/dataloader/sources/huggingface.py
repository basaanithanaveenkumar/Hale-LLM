from __future__ import annotations

from datasets import load_dataset
from loguru import logger
from torch.utils.data import Dataset

from hale_llm.dataloader.context_windows import pack_context_windows


def load_hf_split(cfg, split: str):
    data = cfg.data
    name = data.subset if data.subset else None
    kwargs: dict = {"split": split}
    if data.cache_dir:
        kwargs["cache_dir"] = data.cache_dir
    if name:
        raw = load_dataset(data.dataset, name, **kwargs)
    else:
        raw = load_dataset(data.dataset, **kwargs)
    size = data.train_size if split == data.train_split else data.val_size
    if size is not None:
        raw = raw.select(range(min(size, len(raw))))
    logger.info(
        "hf dataset={} subset={} split={} rows={} text_field={}",
        data.dataset,
        data.subset,
        split,
        len(raw),
        data.text_field,
    )
    return raw


def documents_from_split(raw, text_field: str = "text") -> list[str]:
    """One string per dataset row; empty rows are dropped. Each row is one document."""
    documents = []
    for row in raw:
        text = (row.get(text_field) or "").strip()
        if text:
            documents.append(text)
    return documents


class HuggingFaceTextDataset(Dataset):
    """Hugging Face split packed into non-overlapping context windows (one per item)."""

    def __init__(self, tokenizer, cfg, split: str):
        self.tokenizer = tokenizer
        self.context_window = cfg.model.max_length  # set from data.context_window
        raw = load_hf_split(cfg, split)
        documents = documents_from_split(raw, cfg.data.text_field)
        self.ids = pack_context_windows(tokenizer, documents, self.context_window)
        logger.info(
            "split={} documents={} -> {:,} context windows of {} tokens ({:,} tokens)",
            split,
            len(documents),
            len(self.ids),
            self.context_window,
            self.ids.numel(),
        )

    def __len__(self):
        return int(self.ids.size(0))

    def __getitem__(self, idx):
        return self.ids[idx]
