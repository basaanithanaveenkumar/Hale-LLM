"""Name -> router lookup (``model.moe_router`` in YAML)."""

from __future__ import annotations

from collections.abc import Callable

from torch import nn

from hale_llm.registry import NamedRegistry

ROUTERS = NamedRegistry("router")


def register_router(name: str) -> Callable[[type], type]:
    """Class decorator: make a router selectable from the config."""

    def decorator(cls: type) -> type:
        ROUTERS.add(name, cls)
        return cls

    return decorator


def build_router(name: str, d_model: int, n_experts: int, top_k: int) -> nn.Module:
    return ROUTERS.get(name)(d_model, n_experts, top_k)
