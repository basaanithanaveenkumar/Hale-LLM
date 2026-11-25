"""Mix windows from naturally long documents into the normal windows."""

from __future__ import annotations

import torch
from loguru import logger

from hale_llm.dataloader.context_windows import pack_token_documents


def mix_long_documents(
    token_documents: list[list[int]],
    eos_id: int,
    context_window: int,
    long_fraction: float,
    min_long_tokens: int,
) -> torch.Tensor:
    """Windows where about ``long_fraction`` come from long documents and the rest from everything.

    Why: extending the context only on long texts makes a model forget short ones.
    Real long documents (books, papers, whole code files) are the only data where the
    far end of a long window carries information from the near end; padding a window
    with unrelated short stories teaches nothing about long range.

    Steps:

    1. pack all documents as usual: the "regular" windows;
    2. pack only the documents with at least ``min_long_tokens`` tokens: the "long" windows
       (mostly one document each, because a long document fills a window by itself);
    3. swap ``long_fraction`` of the regular windows for long ones, keeping the total
       number of windows the same. If there are too few long windows, use all of them and warn.
    """
    if not 0.0 <= long_fraction < 1.0:
        raise ValueError(f"long_fraction must be in [0, 1), got {long_fraction}")
    regular = pack_token_documents(token_documents, eos_id, context_window)
    if long_fraction == 0.0 or len(regular) == 0:
        return regular

    long_documents = [ids for ids in token_documents if len(ids) + 1 >= min_long_tokens]
    long_windows = pack_token_documents(long_documents, eos_id, context_window)
    wanted = int(long_fraction * len(regular))
    available = len(long_windows)
    if available < wanted:
        logger.warning(
            "only {} long-document windows (>= {} tokens per document) for the {} wanted "
            "({:.0%} of {}); using all of them",
            available,
            min_long_tokens,
            wanted,
            long_fraction,
            len(regular),
        )
    count = min(wanted, available)
    if count == 0:
        return regular

    # Spread the chosen long windows evenly over the available ones.
    picks = torch.linspace(0, available - 1, count).round().long()
    mixed = torch.cat([regular[: len(regular) - count], long_windows[picks]], dim=0)
    logger.info("window mix: {} regular + {} from long documents", len(regular) - count, count)
    return mixed
