"""Name -> vocabulary-size-policy lookup (``tokenizer.vocab_policy`` in YAML)."""

from __future__ import annotations

from collections.abc import Callable

from hale_core.registry import NamedRegistry

from hale_llm.tokenizer.vocab.vocab_size_policy import VocabSizePolicy

VOCAB_POLICIES = NamedRegistry("vocab_policy")


def register_vocab_policy(name: str) -> Callable[[type[VocabSizePolicy]], type[VocabSizePolicy]]:
    """Class decorator: make ``name`` selectable as ``tokenizer.vocab_policy``."""

    def decorator(cls: type[VocabSizePolicy]) -> type[VocabSizePolicy]:
        cls.name = name
        VOCAB_POLICIES.add(name, cls)
        return cls

    return decorator


def get_vocab_policy(name: str) -> VocabSizePolicy:
    """Instantiate the policy registered under ``name`` (KeyError lists the options)."""
    return VOCAB_POLICIES.get(name)()
