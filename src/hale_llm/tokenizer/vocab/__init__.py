"""Vocabulary-size policies. Importing this package registers ``by_model_size``."""

from hale_llm.tokenizer.vocab.model_size_policy import ModelSizeVocabPolicy
from hale_llm.tokenizer.vocab.param_estimate import estimate_non_embedding_params
from hale_llm.tokenizer.vocab.registry import VOCAB_POLICIES, get_vocab_policy, register_vocab_policy
from hale_llm.tokenizer.vocab.vocab_size_policy import VocabSizePolicy

__all__ = [
    "VOCAB_POLICIES",
    "ModelSizeVocabPolicy",
    "VocabSizePolicy",
    "estimate_non_embedding_params",
    "get_vocab_policy",
    "register_vocab_policy",
]
