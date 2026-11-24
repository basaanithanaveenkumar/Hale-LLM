"""Positional-encoding schemes. Importing this package registers ``rope`` and ``learned``."""

from hale_llm.nn.positional.learned import LearnedPositionalEncoding
from hale_llm.nn.positional.positional_encoding import PositionalEncoding
from hale_llm.nn.positional.registry import (
    POSITIONAL_ENCODINGS,
    build_positional_encoding,
    register_positional_encoding,
)
from hale_llm.nn.positional.rotary import RotaryPositionalEncoding

__all__ = [
    "POSITIONAL_ENCODINGS",
    "LearnedPositionalEncoding",
    "PositionalEncoding",
    "RotaryPositionalEncoding",
    "build_positional_encoding",
    "register_positional_encoding",
]
