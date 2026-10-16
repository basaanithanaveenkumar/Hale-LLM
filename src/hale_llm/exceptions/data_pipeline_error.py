"""Training or evaluation data cannot be built from the config."""

from __future__ import annotations

from hale_llm.exceptions.hale_llm_error import HaleLLMError


class DataPipelineError(HaleLLMError, ValueError):
    """Training or evaluation data cannot be built from the config.

    For example ``data.source: overfit`` without ``data.overfit_text`` or a window stride below 1.
    Also a :class:`ValueError`.
    """
