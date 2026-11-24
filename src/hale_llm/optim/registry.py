"""Optimizers owned by hale_llm, looked up before the hale_core ones."""

from __future__ import annotations

from collections.abc import Callable

from hale_core.registry import NamedRegistry

OPTIMIZERS = NamedRegistry("optimizer")


def register_optimizer(name: str) -> Callable[[type], type]:
    """Class decorator: make ``name`` selectable as ``train.optimizer_type``."""

    def decorator(cls: type) -> type:
        OPTIMIZERS.add(name.lower(), cls)
        return cls

    return decorator
