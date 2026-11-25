"""Tokenizer setup shared by training, evaluation and inference."""

from __future__ import annotations

from functools import lru_cache

from transformers import AutoTokenizer


@lru_cache(maxsize=8)
def setup_tokenizer(name: str = "gpt2"):
    """Load a Hugging Face tokenizer and make it usable for batching.

    GPT-2 has no pad token and none is added: packed windows are always full, and
    its end-of-text token is kept as a real stop token (see ``dataloader.padding``).

    Args:
        name: tokenizer id on the Hub (``data.tokenizer_name``).

    The result is cached, so the several places that need a tokenizer in one run
    (data module, loaders, inference) share one object and the Hub is read once.
    """
    return AutoTokenizer.from_pretrained(name)


_LOADED: dict[tuple, object] = {}  # in-process memo; the files on disk are the source of truth


def build_tokenizer_for(cfg, *, allow_training: bool = True):
    """The tokenizer for a run.

    ``tokenizer.source: trained`` (default): load the run's trained tokenizer, or train
    one on the training data if none exists yet (``allow_training=False`` forbids that,
    used when loading a model for inference). ``pretrained``: a Hub tokenizer such as gpt2.
    """
    if cfg.tokenizer.source == "pretrained":
        return setup_tokenizer(cfg.data.tokenizer_name)

    from hale_llm.tokenizer import TokenizerBuilder

    t = cfg.tokenizer
    key = (t.path, t.kind, t.vocab_size, cfg.data.dataset)
    if key not in _LOADED:
        _LOADED[key] = TokenizerBuilder(cfg).build(allow_training=allow_training)
    return _LOADED[key]
