"""Interface every tokenizer-training algorithm implements."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from tokenizers import Tokenizer


class TokenizerTrainer(ABC):
    """Learns a byte-level tokenizer from raw documents.

    A new algorithm is one subclass plus ``@register_tokenizer_trainer("name")``.
    """

    #: Name used in ``tokenizer.kind``.
    name: str = "base"

    @abstractmethod
    def train(
        self,
        documents: Sequence[str],
        vocab_size: int,
        special_tokens: Sequence[str],
    ) -> Tokenizer:
        """Return a trained ``tokenizers.Tokenizer`` with at most ``vocab_size`` entries.

        ``special_tokens`` get the lowest ids, in the order given.
        """
