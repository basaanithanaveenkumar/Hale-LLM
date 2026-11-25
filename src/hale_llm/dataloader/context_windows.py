"""Pack documents into fixed-length, non-overlapping context windows."""

from __future__ import annotations

from collections.abc import Iterable

import torch


def tokenize_documents(tokenizer, documents: Iterable[str]) -> list[list[int]]:
    """One id list per non-empty document (no special tokens added)."""
    token_documents: list[list[int]] = []
    for text in documents:
        text = text.strip()
        if not text:
            continue  # empty rows carry no text; skip them rather than emit a bare EOS
        ids = tokenizer(
            text,
            add_special_tokens=False,
            truncation=False,
            padding=False,
            return_tensors=None,
        )["input_ids"]
        token_documents.append(list(ids))
    return token_documents


def join_with_eos(token_documents: list[list[int]], eos_id: int) -> list[int]:
    """Concatenate the documents with an EOS token after each one.

    Tiny example with EOS = 9: ``[[a, b], [c]]`` -> ``[a, b, 9, c, 9]``.
    The EOS token is the stop token: it marks where one document ends and the next
    begins, so the model learns to end a text instead of rambling into the next one.
    No BOS token is added; GPT-2 (and gpt-oss at generation time) start a text with
    the first real token.
    """
    stream: list[int] = []
    for ids in token_documents:
        stream.extend(ids)
        stream.append(eos_id)
    return stream


def require_eos_id(tokenizer) -> int:
    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        raise ValueError("tokenizer has no EOS token; cannot mark document boundaries")
    return eos_id


def encode_documents(tokenizer, documents: Iterable[str]) -> list[int]:
    """Tokenise each document and join them into one id stream, EOS after every document."""
    eos_id = require_eos_id(tokenizer)
    return join_with_eos(tokenize_documents(tokenizer, documents), eos_id)


def pack_token_documents(token_documents: list[list[int]], eos_id: int, context_window: int) -> torch.Tensor:
    """Cut already-tokenised documents into back-to-back windows of ``context_window`` tokens.

    Each window is a training example. Windows do not overlap, so every token is seen
    exactly once per epoch, and every window is full, so there is no padding. The few
    tokens left over at the end (fewer than one window) are dropped.

    Returns a ``(n_windows, context_window)`` long tensor; ``n_windows`` can be 0.
    """
    if context_window < 2:
        raise ValueError(f"context_window must be >= 2, got {context_window}")
    stream = join_with_eos(token_documents, eos_id)
    n_windows = len(stream) // context_window
    if n_windows == 0:
        return torch.zeros(0, context_window, dtype=torch.long)
    flat = torch.tensor(stream[: n_windows * context_window], dtype=torch.long)
    return flat.view(n_windows, context_window)


def pack_context_windows(tokenizer, documents: Iterable[str], context_window: int) -> torch.Tensor:
    """Tokenise ``documents`` and pack them into windows (see :func:`pack_token_documents`)."""
    eos_id = require_eos_id(tokenizer)
    return pack_token_documents(tokenize_documents(tokenizer, documents), eos_id, context_window)
