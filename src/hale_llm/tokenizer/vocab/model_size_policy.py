"""Vocabulary sized to the model: small models get small vocabularies."""

from __future__ import annotations

from hale_llm.tokenizer.vocab.registry import register_vocab_policy
from hale_llm.tokenizer.vocab.vocab_size_policy import VocabSizePolicy

# (upper bound on non-embedding parameters, vocabulary size). The last row has no bound.
# Why: the embedding table costs vocab x d_model parameters. In SmolLM2-135M it is more
# than 28M of 135M (about 21%); a big vocabulary starves a small model of useful depth.
#   under 500M  -> 32k   (the 32-50k guideline; 2^15, a multiple of 128 for GPU kernels)
#   0.5B - 3B   -> 64k   (2^16)
#   above 3B    -> ~100k (784 x 128)
# The 100k tier is part of the policy but is only reached by models above 3B parameters.
_TIERS: tuple[tuple[int | None, int], ...] = (
    (500_000_000, 32_768),
    (3_000_000_000, 65_536),
    (None, 100_352),
)


@register_vocab_policy("by_model_size")
class ModelSizeVocabPolicy(VocabSizePolicy):
    """Picks the first tier whose parameter bound the model fits under."""

    def choose(self, non_embedding_params: int) -> int:
        if non_embedding_params < 0:
            raise ValueError(f"parameter count must be >= 0, got {non_embedding_params}")
        for bound, vocab_size in _TIERS:
            if bound is None or non_embedding_params < bound:
                return vocab_size
        raise AssertionError("unreachable: the last tier has no bound")  # pragma: no cover
