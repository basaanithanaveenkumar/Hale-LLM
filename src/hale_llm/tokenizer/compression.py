"""How well a tokenizer compresses text."""

from __future__ import annotations

from collections.abc import Sequence


def bytes_per_token(tokenizer, texts: Sequence[str]) -> float:
    """Average UTF-8 bytes covered by one token (higher = fewer tokens for the same text).

    It is also the exchange rate between models with different tokenizers: a loss in
    nats per token divided by ``bytes_per_token`` gives nats per byte, which is the fair
    way to compare a BPE model with a SuperBPE model.
    """
    total_bytes = 0
    total_tokens = 0
    for text in texts:
        if not text:
            continue
        total_bytes += len(text.encode("utf-8"))
        total_tokens += len(tokenizer(text, add_special_tokens=False)["input_ids"])
    if total_tokens == 0:
        raise ValueError("no text to measure")
    return total_bytes / total_tokens
