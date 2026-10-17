"""Tokenizer setup shared by training, evaluation and inference."""

from loguru import logger
from transformers import AutoTokenizer


def setup_tokenizer(name: str = "gpt2"):
    """Load a Hugging Face tokenizer with a pad token and a ``[MASK]`` token.

    Padding reuses EOS (GPT-2 has none); masked-diffusion variants need a dedicated mask token, so
    ``[MASK]`` is added when missing (the model's embedding is sized from ``len(tokenizer)``).

    Args:
        name: Hugging Face tokenizer id.
    """
    logger.debug("loading tokenizer {}", name)
    tok = AutoTokenizer.from_pretrained(name)
    tok.pad_token = tok.eos_token
    if tok.mask_token is None:
        tok.add_special_tokens({"mask_token": "[MASK]"})
        logger.info("added [MASK] token to tokenizer {}", name)
    logger.debug("tokenizer {} vocab_size={}", name, len(tok))
    return tok
