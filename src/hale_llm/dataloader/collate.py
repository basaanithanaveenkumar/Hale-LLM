"""Registry of per-variant collate functions (raw token windows → the batch dict a loss expects)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from hale_llm.core.registry import NamedRegistry

COLLATE_REGISTRY = NamedRegistry("collate")
COLLATES: dict[str, Callable[..., dict[str, Any]]] = COLLATE_REGISTRY.items


def register_collate(name: str):
    """Decorator registering a collate function for variant ``name``.

    Args:
        name: variant name (``"diffusion"`` …).

    Raises:
        hale_core.exceptions.RegistryError: if ``name`` already has a collate function.
    """
    def deco(fn):
        COLLATE_REGISTRY.add(name, fn)
        return fn

    return deco


def get_collate(name: str):
    """The collate function registered for variant ``name``.

    Raises:
        hale_core.exceptions.RegistryError: for an unknown variant (also a ``KeyError``).
    """
    return COLLATE_REGISTRY.get(name)
