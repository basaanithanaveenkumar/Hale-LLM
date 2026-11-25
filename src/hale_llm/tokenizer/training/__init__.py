"""Tokenizer training algorithms. Importing this package registers ``bpe`` and ``superbpe``."""

from hale_llm.tokenizer.training.bpe_tokenizer_trainer import BpeTokenizerTrainer
from hale_llm.tokenizer.training.pair_merger import PairMerger
from hale_llm.tokenizer.training.registry import (
    TOKENIZER_TRAINERS,
    build_tokenizer_trainer,
    register_tokenizer_trainer,
)
from hale_llm.tokenizer.training.superbpe_tokenizer_trainer import SuperBpeTokenizerTrainer
from hale_llm.tokenizer.training.tokenizer_trainer import TokenizerTrainer

__all__ = [
    "TOKENIZER_TRAINERS",
    "BpeTokenizerTrainer",
    "PairMerger",
    "SuperBpeTokenizerTrainer",
    "TokenizerTrainer",
    "build_tokenizer_trainer",
    "register_tokenizer_trainer",
]
