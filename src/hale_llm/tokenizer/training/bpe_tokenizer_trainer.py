"""Standard byte-level BPE with the cl100k pre-tokenizer."""

from __future__ import annotations

from collections.abc import Sequence

from loguru import logger
from tokenizers import Tokenizer, models, trainers

from hale_llm.config.sections.tokenizer import TokenizerConfig
from hale_llm.tokenizer.training.byte_level_pipeline import (
    byte_alphabet,
    byte_level_decoder,
    regex_byte_level_pre_tokenizer,
)
from hale_llm.tokenizer.training.registry import register_tokenizer_trainer
from hale_llm.tokenizer.training.tokenizer_trainer import TokenizerTrainer


@register_tokenizer_trainer("bpe")
class BpeTokenizerTrainer(TokenizerTrainer):
    """Byte-pair encoding: start from bytes, repeatedly merge the most frequent adjacent pair.

    Tiny example: in "low low lower", ``l`` + ``o`` is the most frequent pair, so ``lo``
    becomes a token; then ``lo`` + ``w`` gives ``low``. Merges never cross a pre-token
    boundary, so a token is always part of one word (or one punctuation run, one number).
    """

    def __init__(self, settings: TokenizerConfig) -> None:
        self._settings = settings

    def train_stage(
        self, documents: Sequence[str], vocab_size: int, special_tokens: Sequence[str]
    ) -> Tokenizer:
        """Train the pre-tokenized BPE stage. SuperBPE reuses this as its first stage."""
        tokenizer = Tokenizer(models.BPE())
        tokenizer.pre_tokenizer = regex_byte_level_pre_tokenizer()
        tokenizer.decoder = byte_level_decoder()
        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=self._settings.min_frequency,
            special_tokens=list(special_tokens),
            initial_alphabet=byte_alphabet(),
            show_progress=False,
        )
        tokenizer.train_from_iterator(documents, trainer=trainer)
        return tokenizer

    def train(
        self, documents: Sequence[str], vocab_size: int, special_tokens: Sequence[str]
    ) -> Tokenizer:
        logger.info("training byte-level BPE: vocab_size={} on {:,} documents", vocab_size, len(documents))
        tokenizer = self.train_stage(documents, vocab_size, special_tokens)
        logger.info("BPE done: {} tokens", tokenizer.get_vocab_size())
        return tokenizer
