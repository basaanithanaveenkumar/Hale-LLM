"""The training loop cannot continue."""

from __future__ import annotations

from hale_llm.exceptions.hale_llm_error import HaleLLMError


class TrainingError(HaleLLMError, RuntimeError):
    """The training loop cannot continue.

    Raised for an empty dataloader or an invalid schedule. Also a :class:`RuntimeError`, the type
    these sites raised before the hierarchy existed.
    """
