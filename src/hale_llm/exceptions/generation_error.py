"""A sampler was called with inputs it cannot handle."""

from __future__ import annotations

from hale_llm.exceptions.hale_llm_error import HaleLLMError


class GenerationError(HaleLLMError, ValueError):
    """A sampler was called with inputs it cannot handle.

    For example speculative multi-token decoding with a batch larger than one. Also a
    :class:`ValueError`.
    """
