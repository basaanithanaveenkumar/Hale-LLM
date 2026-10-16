"""Root of every error raised on purpose by :mod:`hale_llm`."""

from __future__ import annotations

from hale_core.exceptions import HaleCoreError


class HaleLLMError(HaleCoreError):
    """Root of every error raised on purpose by :mod:`hale_llm`.

    Subclasses :class:`hale_core.exceptions.HaleCoreError`, so one ``except HaleCoreError`` in a CLI
    or job runner handles expected failures from HaleBlocks and every Hale* package alike.
    """
