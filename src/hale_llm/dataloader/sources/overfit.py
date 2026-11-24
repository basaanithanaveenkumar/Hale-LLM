from __future__ import annotations

import torch
from loguru import logger
from torch.utils.data import DataLoader, TensorDataset

from hale_llm.dataloader.collate import get_collate
from hale_llm.dataloader.context_windows import encode_documents, pack_context_windows


def uses_overfit(cfg) -> bool:
    return cfg.data.source == "overfit" or bool(cfg.data.overfit_text)


def overfit_ids(tokenizer, text: str, context_window: int, n_copies: int) -> torch.Tensor:
    """One full context window made of ``text`` repeated (EOS after each copy), tiled ``n_copies`` times."""
    one_copy = len(encode_documents(tokenizer, [text]))
    if one_copy == 0:
        raise ValueError("overfit_text is empty")
    # Enough repetitions to fill at least one window.
    repeats = -(-context_window // one_copy)
    window = pack_context_windows(tokenizer, [text] * repeats, context_window)[0]
    return window.unsqueeze(0).repeat(n_copies, 1)


def make_overfit_loader(
    variant: str,
    tokenizer,
    text: str,
    context_window: int,
    batch_size: int,
    n_copies: int,
    **collate_kwargs,
) -> DataLoader:
    ids = overfit_ids(tokenizer, text, context_window, n_copies)
    ds = TensorDataset(ids)
    collate = get_collate(variant)

    def _collate(batch):
        tensors = torch.stack([b[0] for b in batch], dim=0)
        return collate(tensors, tokenizer=tokenizer, **collate_kwargs)

    logger.info(
        "overfit loader text={!r} copies={} seq_len={} batch_size={}",
        text,
        n_copies,
        context_window,
        min(batch_size, n_copies),
    )
    return DataLoader(
        ds, batch_size=min(batch_size, n_copies), shuffle=True, drop_last=True, collate_fn=_collate
    )
