"""SuperBPE: BPE whose second stage may merge across word boundaries."""

from __future__ import annotations

import json
from collections.abc import Sequence

from loguru import logger
from tokenizers import Tokenizer, models

from hale_llm.config.sections.tokenizer import TokenizerConfig
from hale_llm.tokenizer.training.bpe_tokenizer_trainer import BpeTokenizerTrainer
from hale_llm.tokenizer.training.byte_level_pipeline import (
    byte_level_decoder,
    plain_byte_level_pre_tokenizer,
)
from hale_llm.tokenizer.training.pair_merger import PairMerger
from hale_llm.tokenizer.training.registry import register_tokenizer_trainer
from hale_llm.tokenizer.training.tokenizer_trainer import TokenizerTrainer


@register_tokenizer_trainer("superbpe")
class SuperBpeTokenizerTrainer(TokenizerTrainer):
    """Two-stage training (Liu et al., 2025, "SuperBPE").

    Stage 1 is ordinary BPE with the cl100k pre-tokenizer, up to ``transition_fraction``
    of the vocabulary. Stage 2 drops the pre-tokenizer and keeps merging, so tokens can
    now span spaces and become "superwords" such as " of the" or "in order to".
    Common multi-word phrases then cost one token instead of three or four, so the same
    text needs fewer tokens (fewer forward passes per byte of text).

    Steps:

    1. train stage-1 BPE on the documents (words are the unit);
    2. re-encode a sample of the documents with it;
    3. keep merging the most frequent adjacent token pair, now over whole documents;
    4. join both merge lists into one tokenizer that has no pre-tokenizer.
    """

    def __init__(self, settings: TokenizerConfig) -> None:
        self._settings = settings
        self._stage_one = BpeTokenizerTrainer(settings)

    def train(
        self, documents: Sequence[str], vocab_size: int, special_tokens: Sequence[str]
    ) -> Tokenizer:
        stage_one_size = max(len(special_tokens) + 256, int(vocab_size * self._settings.transition_fraction))
        logger.info(
            "training SuperBPE: vocab_size={} (stage 1 up to {}, stage 2 adds the rest) on {:,} documents",
            vocab_size,
            stage_one_size,
            len(documents),
        )

        # Step 1: ordinary word-level BPE.
        stage_one = self._stage_one.train_stage(documents, stage_one_size, special_tokens)
        vocab, merges = self._export(stage_one)

        # Step 2: encode a sample with stage 1 (the sample size caps memory and time).
        token_documents = self._encode_sample(stage_one, documents)
        if not token_documents:
            logger.warning("no text to learn superword merges from; returning the stage-1 tokenizer")
            return self._assemble(vocab, merges, special_tokens)

        # Step 3: continue merging without word boundaries.
        id_to_token = [token for token, _ in sorted(vocab.items(), key=lambda item: item[1])]
        merger = PairMerger(token_documents, id_to_token, self._settings.min_frequency)
        new_merges = merger.learn(vocab_size - len(vocab))
        logger.info("stage 2 learned {:,} merges over {:,} tokens", len(new_merges), sum(map(len, token_documents)))

        # Step 4: one merge list, stage-1 merges first (their ranks stay lower).
        full_vocab = {token: i for i, token in enumerate(merger.vocabulary)}
        return self._assemble(full_vocab, merges + new_merges, special_tokens)

    @staticmethod
    def _export(tokenizer: Tokenizer) -> tuple[dict[str, int], list[tuple[str, str]]]:
        """Read vocabulary and merge list out of a trained BPE tokenizer."""
        model = json.loads(tokenizer.to_str())["model"]
        merges = [tuple(m.split(" ", 1)) if isinstance(m, str) else tuple(m) for m in model["merges"]]
        return model["vocab"], merges

    def _encode_sample(self, stage_one: Tokenizer, documents: Sequence[str]) -> list[list[int]]:
        """Stage-1 ids for documents, stopping once ``stage2_max_tokens`` tokens are collected."""
        limit = self._settings.stage2_max_tokens
        collected: list[list[int]] = []
        total = 0
        chunk = 2_000
        for start in range(0, len(documents), chunk):
            for encoding in stage_one.encode_batch(list(documents[start : start + chunk])):
                if encoding.ids:
                    collected.append(encoding.ids)
                    total += len(encoding.ids)
            if total >= limit:
                break
        return collected

    @staticmethod
    def _assemble(
        vocab: dict[str, int], merges: list[tuple[str, str]], special_tokens: Sequence[str]
    ) -> Tokenizer:
        """BPE model over the whole text: no pre-tokenizer, so merges can cross spaces."""
        tokenizer = Tokenizer(models.BPE(vocab=vocab, merges=merges, cache_capacity=0))
        tokenizer.pre_tokenizer = plain_byte_level_pre_tokenizer()
        tokenizer.decoder = byte_level_decoder()
        tokenizer.add_special_tokens(list(special_tokens))
        return tokenizer
