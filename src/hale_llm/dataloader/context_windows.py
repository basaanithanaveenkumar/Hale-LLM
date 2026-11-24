"""Pack documents into fixed-length, non-overlapping context windows."""

from __future__ import annotations

from collections.abc import Iterable

import torch


def encode_documents(tokenizer, documents: Iterable[str]) -> list[int]:
    """Tokenise each document and join them into one id stream, EOS after every document.

    Tiny example with EOS = 9: documents ``["a b", "c"]`` -> ``[a, b, 9, c, 9]``.
    The EOS token is the stop token: it marks where one document ends and the next
    begins, so the model learns to end a text instead of rambling into the next one.
    No BOS token is added; GPT-2 (and gpt-oss at generation time) start a text with
    the first real token.
    """
    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        raise ValueError("tokenizer has no EOS token; cannot mark document boundaries")
    stream: list[int] = []
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
        stream.extend(ids)
        stream.append(eos_id)
    return stream


def pack_context_windows(tokenizer, documents: Iterable[str], context_window: int) -> torch.Tensor:
    """Cut the document stream into back-to-back windows of ``context_window`` tokens.

    Each window is a training example. Windows do not overlap, so every token is seen
    exactly once per epoch, and every window is full, so there is no padding. The few
    tokens left over at the end (fewer than one window) are dropped.

    Returns a ``(n_windows, context_window)`` long tensor; ``n_windows`` can be 0.
    """
    if context_window < 2:
        raise ValueError(f"context_window must be >= 2, got {context_window}")
    stream = encode_documents(tokenizer, documents)
    n_windows = len(stream) // context_window
    if n_windows == 0:
        return torch.zeros(0, context_window, dtype=torch.long)
    flat = torch.tensor(stream[: n_windows * context_window], dtype=torch.long)
    return flat.view(n_windows, context_window)
