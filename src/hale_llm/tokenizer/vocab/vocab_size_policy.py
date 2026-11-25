"""Interface for rules that choose a vocabulary size."""

from __future__ import annotations

from abc import ABC, abstractmethod


class VocabSizePolicy(ABC):
    """Maps a model's size to the vocabulary it should use."""

    #: Name used in ``tokenizer.vocab_policy``.
    name: str = "base"

    @abstractmethod
    def choose(self, non_embedding_params: int) -> int:
        """Return a vocabulary size for a model with this many non-embedding parameters."""
