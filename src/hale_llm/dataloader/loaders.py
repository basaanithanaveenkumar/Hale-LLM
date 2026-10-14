from __future__ import annotations

import hashlib
import json
import re

import torch
import torch.distributed as dist
from hale_core.data.webdataset import WebDatasetAdapter
from loguru import logger
from torch.utils.data import DataLoader, IterableDataset
from torch.utils.data.distributed import DistributedSampler

from hale_llm.dataloader.collate import get_collate
from hale_llm.dataloader.sources.huggingface import HuggingFaceTextDataset
from hale_llm.dataloader.tokenizer import setup_tokenizer


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
        "tokenizer": data.tokenizer_name,
        "max_length": cfg.model.max_length,
        "stride_words": data.stride_words,
        "add_special_tokens": data.add_special_tokens,
    }
    return hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()


def webdataset_name(cfg, hf_split: str, fingerprint: str) -> str:
    """Shard directory name: readable dataset/split plus a short fingerprint, so different data
    configs never overwrite each other's shards."""
    base = f"{cfg.data.dataset}-{cfg.data.subset or 'default'}-{hf_split}"
    return f"{re.sub(r'[^A-Za-z0-9._-]+', '_', base)}-{fingerprint[:12]}"


def make_dataloader(cfg, split: str = "train", **collate_kwargs) -> DataLoader:
    tok = setup_tokenizer(cfg.data.tokenizer_name)
    hf_split = cfg.data.train_split if split == "train" else cfg.data.val_split
    training = split == "train"

    # With data.webdataset.enabled the HF download + tokenisation only runs when the shards are
    # missing or stale (the factory is not called otherwise); with the flag off this is the
    # plain HuggingFaceTextDataset as before.
    fingerprint = dataset_fingerprint(cfg, hf_split)
    ds = WebDatasetAdapter(cfg.data.webdataset).prepare(
        lambda: HuggingFaceTextDataset(tok, cfg, split=hf_split),
        webdataset_name(cfg, hf_split, fingerprint),
        training=training,
        fingerprint=fingerprint,
    )
    collate = get_collate(cfg.variant)

    def _collate(batch):
        tensors = torch.stack(list(batch), dim=0)
        return collate(tensors, tokenizer=tok, **collate_kwargs)

    logger.debug(
        "building dataloader split={} batch_size={} variant={} webdataset={}",
        split,
        cfg.train.batch_size,
        cfg.variant,
        cfg.data.webdataset.enabled,
    )
    loader_kwargs: dict = {
        "batch_size": cfg.train.batch_size,
        "drop_last": True,
        "collate_fn": _collate,
    }
    if isinstance(ds, IterableDataset):
        pass  # WebDataset streams shuffle and split across ranks themselves
    elif dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
        # DDP / Ray: every rank reads its own 1/world_size of the windows; the trainer calls
        # sampler.set_epoch(epoch) so the order changes between epochs.
        loader_kwargs["sampler"] = DistributedSampler(ds, shuffle=training, drop_last=True)
    else:
        loader_kwargs["shuffle"] = training
    return DataLoader(ds, **loader_kwargs)
