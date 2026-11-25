"""Name -> normalisation-layer lookup (``model.norm_type`` in YAML)."""

from __future__ import annotations

from collections.abc import Callable

from hale_llm.registry import NamedRegistry
from torch import nn

NORMS = NamedRegistry("norm")


def register_norm(name: str) -> Callable[[type], type]:
    """Class decorator: make ``name`` selectable from the config."""

    def decorator(cls: type) -> type:
        NORMS.add(name, cls)
        return cls

    return decorator


def build_norm(name: str, dim: int) -> nn.Module:
    """Create a norm layer over the last dimension of size ``dim``."""
    return NORMS.get(name)(dim)
