"""Encode a prompt with the session tokenizer (library helper)."""

from __future__ import annotations

import torch
from torch import nn


def encode_prompt(tokenizer, prompt: str, model: nn.Module, device: str) -> torch.Tensor:
    """Token ids ``[1, n]`` of ``prompt``, keeping the *last* ``max_length`` tokens.

    The most recent text is what the next token depends on, so a prompt longer than the
    context window loses its beginning, never its end.
    """
    ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False, padding=False)["input_ids"]
    backbone = getattr(model, "backbone", None)
    max_length = getattr(backbone, "max_length", None)
    if max_length is not None:
        ids = ids[:, -max_length:]
    return ids.to(device)
