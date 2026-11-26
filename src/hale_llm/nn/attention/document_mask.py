"""Attention mask that keeps the documents of a packed window apart."""

from __future__ import annotations

import torch


def cross_document_mask(document_ids: torch.Tensor) -> torch.Tensor:
    """``[B, S]`` document numbers -> ``[B, S, S]`` bool, ``True`` where query ``i`` must NOT see key ``j``.

    Two tokens may attend to each other only if they are in the same document. Every token still sees itself, so
    no row is ever fully blocked. Causality is applied separately by the attention head.
    """
    return document_ids[:, :, None] != document_ids[:, None, :]
