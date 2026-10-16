"""A named or ``latest`` experiment folder does not exist."""

from __future__ import annotations

from hale_llm.exceptions.hale_llm_error import HaleLLMError


class ExperimentNotFoundError(HaleLLMError, FileNotFoundError):
    """A named or ``latest`` experiment folder does not exist.

    Also a :class:`FileNotFoundError`, so ``except FileNotFoundError`` call sites keep working.
    """
