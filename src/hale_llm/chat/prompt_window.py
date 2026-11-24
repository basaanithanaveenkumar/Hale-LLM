"""Keep a growing conversation inside the model's context window."""

from __future__ import annotations

from loguru import logger


class PromptWindow:
    """Trims text from the *left* so that only the newest tokens remain.

    The model can only see ``max_length`` tokens, and part of that budget is
    reserved for the tokens it is about to generate. The default tokenizer
    truncation drops the end of the prompt, which would throw away the user's
    latest message, so the chat layer trims the old context instead.
    """

    def __init__(self, tokenizer, max_tokens: int) -> None:
        if max_tokens < 1:
            raise ValueError(f"max_tokens must be >= 1, got {max_tokens}")
        self._tokenizer = tokenizer
        self._max_tokens = max_tokens

    @property
    def max_tokens(self) -> int:
        return self._max_tokens

    def fit(self, text: str) -> str:
        """Return ``text`` unchanged if it fits, otherwise its last ``max_tokens`` tokens."""
        # Plain list of ids: no padding, no truncation, no special tokens.
        ids = self._tokenizer(
            text,
            add_special_tokens=False,
            padding=False,
            truncation=False,
            return_tensors=None,
        )["input_ids"]
        if len(ids) <= self._max_tokens:
            return text
        logger.debug("trimming prompt from {} to {} tokens", len(ids), self._max_tokens)
        return self._tokenizer.decode(ids[-self._max_tokens :], skip_special_tokens=True)
