"""Tokenizer setup shared by training, evaluation and inference."""

from __future__ import annotations

from functools import lru_cache

from loguru import logger
from transformers import AutoTokenizer


@lru_cache(maxsize=8)
def setup_tokenizer(name: str = "gpt2", add_mask_token: bool = False):
    """Load a Hugging Face tokenizer and make it usable for batching.

    GPT-2 has no pad token and none is added: packed windows are always full, and
    its end-of-text token is kept as a real stop token (see ``dataloader.padding``).

    Args:
        name: tokenizer id on the Hub (``data.tokenizer_name``).
        add_mask_token: add a ``[MASK]`` token if the tokenizer has none. Only the
            diffusion-style paradigms need it, because they corrupt text by replacing
            tokens with ``[MASK]``. An autoregressive model never sees it, and adding
            it would grow the embedding table by one unused row (GPT-2: 50,257 -> 50,258).

    The result is cached, so the several places that need a tokenizer in one run
    (data module, loaders, inference) share one object and the Hub is read once.
    """
    tok = AutoTokenizer.from_pretrained(name)
    if add_mask_token and tok.mask_token is None:
        tok.add_special_tokens({"mask_token": "[MASK]"})
        logger.info("added [MASK] token to tokenizer {}", name)
    return tok


_LOADED: dict[tuple, object] = {}  # in-process memo; the files on disk are the source of truth


def build_tokenizer_for(cfg, *, allow_training: bool = True):
    """The tokenizer for a run.

    ``tokenizer.source: trained`` (default): load the run's trained tokenizer, or train
    one on the training data if none exists yet (``allow_training=False`` forbids that,
    used when loading a model for inference). ``pretrained``: a Hub tokenizer such as gpt2.
    """
    import hale_llm.models  # noqa: F401  (registers the variants)
    from hale_llm.core.registry import get_variant

    needs_mask = getattr(get_variant(cfg.variant), "uses_mask_token", False)
    if cfg.tokenizer.source == "pretrained":
        return setup_tokenizer(cfg.data.tokenizer_name, add_mask_token=needs_mask)

    from hale_llm.tokenizer import TokenizerBuilder

    t = cfg.tokenizer
    key = (t.path, needs_mask, t.kind, t.vocab_size, cfg.data.dataset)
    if key not in _LOADED:
        _LOADED[key] = TokenizerBuilder(cfg, with_mask=needs_mask).build(allow_training=allow_training)
    return _LOADED[key]
