"""Cut generated text at the first stop token."""

from __future__ import annotations

import torch


def cut_at_stop(token_ids: torch.Tensor, prompt_len: int, stop_id: int | None) -> torch.Tensor:
    """Drop everything from the first ``stop_id`` after the prompt onwards.

    Autoregressive sampling stops by itself, but the diffusion samplers always fill
    the whole output length, so the text after an end-of-text token is cut here.

    Args:
        token_ids: 1-D tensor of prompt + generated ids.
        prompt_len: number of leading prompt tokens (a stop token inside the prompt is kept).
        stop_id: the stop token id, or None for tokenizers without one.
    """
    if stop_id is None:
        return token_ids
    generated = token_ids[prompt_len:]
    hits = (generated == stop_id).nonzero()
    if hits.numel() == 0:
        return token_ids
    return token_ids[: prompt_len + int(hits[0])]
