"""An animation or figure cannot be drawn from the given data."""

from __future__ import annotations

from hale_llm.exceptions.hale_llm_error import HaleLLMError


class VisualizationError(HaleLLMError, ValueError):
    """An animation or figure cannot be drawn from the given data.

    Raised for inconsistent inputs such as mismatched list lengths, an empty frame sequence or
    characters outside the explainer vocabulary. Also a :class:`ValueError`.
    """
