"""Encode a prompt with the session tokenizer (library helper)."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn


def encode_prompt(tokenizer, prompt: str, model: nn.Module, device: str) -> torch.Tensor:
    """Tokenise ``prompt`` (no special tokens), truncated to the backbone's ``max_length``.

    Returns:
        ``(1, L)`` token ids on ``device``.
    """
    encode_kw: dict[str, Any] = {
        "return_tensors": "pt",
        "add_special_tokens": False,
        "padding": False,
        "truncation": True,
    }
    backbone = getattr(model, "backbone", None)
    if backbone is not None and hasattr(backbone, "max_length"):
        encode_kw["max_length"] = backbone.max_length
    return tokenizer(prompt, **encode_kw)["input_ids"].to(device)
