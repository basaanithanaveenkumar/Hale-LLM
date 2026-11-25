"""Name -> tokenizer-trainer lookup (``tokenizer.kind`` in YAML)."""

from __future__ import annotations

from collections.abc import Callable

from hale_llm.registry import NamedRegistry

from hale_llm.config.sections.tokenizer import TokenizerConfig
from hale_llm.tokenizer.training.tokenizer_trainer import TokenizerTrainer

TOKENIZER_TRAINERS = NamedRegistry("tokenizer_trainer")


def register_tokenizer_trainer(name: str) -> Callable[[type[TokenizerTrainer]], type[TokenizerTrainer]]:
    """Class decorator: make ``name`` selectable as ``tokenizer.kind``."""

    def decorator(cls: type[TokenizerTrainer]) -> type[TokenizerTrainer]:
        cls.name = name
        TOKENIZER_TRAINERS.add(name, cls)
        return cls

    return decorator


def build_tokenizer_trainer(settings: TokenizerConfig) -> TokenizerTrainer:
    """Instantiate the trainer named by ``settings.kind`` with its settings."""
    return TOKENIZER_TRAINERS.get(settings.kind)(settings)
