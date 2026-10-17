"""Word-stride packing of raw text into (N, max_length) token windows."""

from __future__ import annotations

import torch

from hale_llm.exceptions import DataPipelineError


def encode_words(tokenizer, words: list[str]) -> tuple[list[int], list[int]]:
    """Tokenise ``words`` (space-prefixed after the first) and record where each word starts.

    Returns:
        ``(ids, starts)``: the flat token ids and the index in ``ids`` of each word's first token.
        Words that tokenise to nothing get the unknown (or EOS) id so positions stay aligned.
    """
    ids: list[int] = []
    starts: list[int] = []
    unk = getattr(tokenizer, "unk_token_id", None) or getattr(tokenizer, "eos_token_id", 0) or 0
    for i, word in enumerate(words):
        starts.append(len(ids))
        piece = word if i == 0 else f" {word}"
        toks = tokenizer.encode(piece, add_special_tokens=False)
        if not toks:
            toks = [unk]
        ids.extend(toks)
    return ids, starts


def sliding_windows(
    tokenizer,
    text: str,
    max_length: int,
    stride_words: int = 10,
) -> torch.Tensor:
    """Pack ``text`` into ``(N, max_length)`` windows starting every ``stride_words`` words.

    Windows always start on a word boundary; the last one is right-padded with ``pad_token_id``.
    Complexity is O(T) in the number of tokens.

    Args:
        tokenizer: Hugging Face tokenizer.
        text: the corpus.
        max_length: tokens per window.
        stride_words: words between consecutive window starts.

    Raises:
        DataPipelineError: if ``stride_words < 1``.
    """
    if stride_words < 1:
        raise DataPipelineError(f"stride_words must be >= 1, got {stride_words}")
    words = text.split()
    if not words:
        return torch.zeros(0, max_length, dtype=torch.long)
    pad_id = int(getattr(tokenizer, "pad_token_id", 0) or 0)
    ids, starts = encode_words(tokenizer, words)
    n_words = len(words)
    rows: list[list[int]] = []
    i = 0
    while i < n_words:
        start = starts[i]
        if start >= len(ids):
            break
        chunk = ids[start : start + max_length]
        if not chunk:
            break
        if len(chunk) < max_length:
            chunk = chunk + [pad_id] * (max_length - len(chunk))
        rows.append(chunk)
        if start + max_length >= len(ids):
            break
        i += stride_words
    if not rows:
        return torch.zeros(0, max_length, dtype=torch.long)
    return torch.tensor(rows, dtype=torch.long)
