"""Small tensor and module helpers shared by training, evaluation and inference."""

from __future__ import annotations

from typing import Any

import torch
from loguru import logger
from torch import nn

# Wrapper modules whose children are listed one level up in the summary table.
_UNWRAP = {"backbone", "model", "module"}


def move_batch_to_device(batch: dict[str, Any], device: str | torch.device) -> dict[str, Any]:
    """Move every tensor in ``batch`` to ``device``; other values pass through."""
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}


def count_parameters(model: nn.Module, *, trainable_only: bool = False) -> int:
    params = model.parameters()
    if trainable_only:
        return sum(p.numel() for p in params if p.requires_grad)
    return sum(p.numel() for p in params)


def _module_row(name: str, module: nn.Module) -> tuple[str, int, int]:
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return name, total, trainable


def _parameter_row(name: str, param: nn.Parameter) -> tuple[str, int, int]:
    return name, param.numel(), param.numel() if param.requires_grad else 0


def _summary_rows(model: nn.Module) -> list[tuple[str, int, int]]:
    rows = [_parameter_row(n, p) for n, p in model.named_parameters(recurse=False)]
    for name, child in model.named_children():
        grandchildren = list(child.named_children())
        if name in _UNWRAP and grandchildren:
            rows += [_parameter_row(f"{name}.{n}", p) for n, p in child.named_parameters(recurse=False)]
            rows += [_module_row(f"{name}.{n}", g) for n, g in grandchildren]
        else:
            rows.append(_module_row(name, child))
    return rows


def format_model_summary(model: nn.Module, *, title: str | None = None) -> str:
    """A fixed-width table of parameter counts per top-level module."""
    name_w, num_w = 32, 16
    bar = "-" * (name_w + num_w * 2 + 2)
    lines = [title] if title else []
    lines += [bar, f"{'Module':<{name_w}}{'Total':>{num_w}}{'Trainable':>{num_w}}", bar]
    for name, total, trainable in _summary_rows(model):
        lines.append(f"{name:<{name_w}}{total / 1e6:>{num_w - 1}.2f}M{trainable / 1e6:>{num_w - 1}.2f}M")
    total = count_parameters(model)
    trainable = count_parameters(model, trainable_only=True)
    lines += [bar, f"{'TOTAL':<{name_w}}{total / 1e6:>{num_w - 1}.2f}M{trainable / 1e6:>{num_w - 1}.2f}M", bar]
    return "\n".join(lines)


def log_model_summary(model: nn.Module, *, title: str | None = None) -> str:
    text = format_model_summary(model, title=title)
    logger.info("\n{}", text)
    return text
