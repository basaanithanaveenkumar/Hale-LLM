"""Name -> RoPE scaling strategy lookup (``model.rope_scaling`` in YAML)."""

from __future__ import annotations

from collections.abc import Callable

from hale_core.registry import NamedRegistry

from hale_llm.nn.positional.scaling.rope_scaling import RopeScaling

ROPE_SCALINGS = NamedRegistry("rope_scaling")


def register_rope_scaling(name: str) -> Callable[[type[RopeScaling]], type[RopeScaling]]:
    """Class decorator: make ``name`` selectable as ``model.rope_scaling``."""

    def decorator(cls: type[RopeScaling]) -> type[RopeScaling]:
        cls.name = name
        ROPE_SCALINGS.add(name, cls)
        return cls

    return decorator


def get_rope_scaling(name: str) -> RopeScaling:
    """Instantiate the strategy registered under ``name`` (KeyError lists the options)."""
    return ROPE_SCALINGS.get(name)()
