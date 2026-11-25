"""Create the tokenizer for a run: train it on the run's data, or load the one already trained."""

from __future__ import annotations

from collections.abc import Sequence

from loguru import logger
from transformers import PreTrainedTokenizerFast

from hale_llm.config.schema import RunConfig
from hale_llm.tokenizer.compression import bytes_per_token
from hale_llm.tokenizer.special_tokens import EOS_TOKEN, MASK_TOKEN
from hale_llm.tokenizer.store import TokenizerStore
from hale_llm.tokenizer.training import build_tokenizer_trainer
from hale_llm.tokenizer.vocab import estimate_non_embedding_params, get_vocab_policy


class TokenizerBuilder:
    """Decides between loading and training, and records what it did.

    By default (``tokenizer.source: trained``) every new run trains its own tokenizer on
    its own training split, so the vocabulary fits the data instead of someone else's.
    The result is saved in the run folder; resuming, evaluating, sampling and chatting
    load that file instead of training again, so a checkpoint always meets the exact
    tokenizer it was trained with.
    """

    def __init__(self, cfg: RunConfig, *, with_mask: bool) -> None:
        self._cfg = cfg
        self._with_mask = with_mask
        self._settings = cfg.tokenizer

    def resolve_vocab_size(self) -> int:
        """Explicit ``tokenizer.vocab_size``, or the policy's choice for this model size."""
        if self._settings.vocab_size is not None:
            return self._settings.vocab_size
        params = estimate_non_embedding_params(self._cfg.model)
        size = get_vocab_policy(self._settings.vocab_policy).choose(params)
        logger.info(
            "vocabulary size {:,} chosen by '{}' for ~{:.1f}M non-embedding parameters",
            size,
            self._settings.vocab_policy,
            params / 1e6,
        )
        return size

    def build(self, *, allow_training: bool) -> PreTrainedTokenizerFast:
        """Load the run's tokenizer; train it first when missing and ``allow_training``."""
        directory = self._settings.path
        if TokenizerStore.exists(directory):
            tokenizer = TokenizerStore.load(directory)
            self._check_mask(tokenizer, directory)
            logger.info("loaded tokenizer from {} ({:,} tokens)", directory, len(tokenizer))
            return tokenizer
        if not allow_training:
            raise FileNotFoundError(
                f"no trained tokenizer at {directory}. Train first, point tokenizer.path at the "
                "run's tokenizer folder, or set tokenizer.source: pretrained"
            )
        return self._train_and_save(directory)

    def _train_and_save(self, directory: str) -> PreTrainedTokenizerFast:
        documents = self.documents(self._cfg.data.train_split)
        vocab_size = self.resolve_vocab_size()
        specials = [EOS_TOKEN] + ([MASK_TOKEN] if self._with_mask else [])
        raw = build_tokenizer_trainer(self._settings).train(documents, vocab_size, specials)
        tokenizer = TokenizerStore.wrap(raw, with_mask=self._with_mask)

        sample = documents[:2000]
        ratio = bytes_per_token(tokenizer, sample)
        logger.info(
            "trained {} tokenizer: {:,} tokens, {:.2f} bytes per token on a sample",
            self._settings.kind,
            len(tokenizer),
            ratio,
        )
        TokenizerStore.save(
            tokenizer,
            directory,
            {
                "kind": self._settings.kind,
                "vocab_size": len(tokenizer),
                "requested_vocab_size": vocab_size,
                "bytes_per_token": ratio,
                "documents": len(documents),
                "dataset": self._cfg.data.dataset,
            },
        )
        return tokenizer

    def documents(self, split: str) -> Sequence[str]:
        """The documents of one split (the training split is what the tokenizer learns from)."""
        # Imported here: the dataloader package imports this module while it initialises.
        from hale_llm.dataloader.sources.huggingface import documents_from_split, load_hf_split
        from hale_llm.dataloader.sources.overfit import uses_overfit

        if uses_overfit(self._cfg):
            return [self._cfg.data.overfit_text or ""]
        documents = documents_from_split(load_hf_split(self._cfg, split), self._cfg.data.text_field)
        if not documents:
            raise ValueError(f"no text in split {split!r} to learn a tokenizer from")
        return documents

    def _check_mask(self, tokenizer: PreTrainedTokenizerFast, directory: str) -> None:
        """A loaded tokenizer must have [MASK] exactly when the variant corrupts text with it."""
        if self._with_mask and tokenizer.mask_token is None:
            raise ValueError(
                f"tokenizer in {directory} has no {MASK_TOKEN} token but variant "
                f"{self._cfg.variant!r} needs one; train a new run"
            )
