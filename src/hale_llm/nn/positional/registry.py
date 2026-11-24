"""Name -> positional-encoding class lookup (``model.pos_encoding`` in YAML)."""

from __future__ import annotations

from collections.abc import Callable

from hale_core.registry import NamedRegistry

from hale_llm.nn.positional.positional_encoding import PositionalEncoding

POSITIONAL_ENCODINGS = NamedRegistry("positional_encoding")


def register_positional_encoding(name: str) -> Callable[[type], type]:
    """Class decorator: make ``name`` selectable from the config."""

    def decorator(cls: type) -> type:
        POSITIONAL_ENCODINGS.add(name, cls)
        return cls

    return decorator


def build_positional_encoding(name: str, **kwargs) -> PositionalEncoding:
    """Instantiate the registered scheme; unknown names raise a KeyError listing the options.

    Each scheme takes the keyword arguments it needs and ignores the rest, so the
    caller can pass one shared set (``head_dim``, ``max_length``, ``d_model``, ...).
    """
    return POSITIONAL_ENCODINGS.get(name)(**kwargs)
