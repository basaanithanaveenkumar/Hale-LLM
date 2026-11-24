"""Padding helpers that also work when the tokenizer has no pad token.

Packed training windows are always full, so real data never contains padding and the
GPT-2 tokenizer is left without a pad token (reusing EOS as pad would make the model
ignore the very stop token it is supposed to learn). Code that builds masks goes
through these helpers so ``pad_id=None`` means "nothing is padding".
"""

from __future__ import annotations

import torch


def pad_positions(input_ids: torch.Tensor, pad_id: int | None) -> torch.Tensor:
    """Boolean mask, True where ``input_ids`` is padding (all False when there is no pad id)."""
    if pad_id is None:
        return torch.zeros_like(input_ids, dtype=torch.bool)
    return input_ids == pad_id


def attention_mask_for(input_ids: torch.Tensor, pad_id: int | None) -> torch.Tensor:
    """1 for real tokens, 0 for padding."""
    return (~pad_positions(input_ids, pad_id)).long()
