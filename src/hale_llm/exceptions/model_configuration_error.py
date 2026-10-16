"""A model config is incompatible with the chosen variant."""

from __future__ import annotations

from hale_llm.exceptions.hale_llm_error import HaleLLMError


class ModelConfigurationError(HaleLLMError, ValueError):
    """A model config is incompatible with the chosen variant.

    Raised at construction time, before any compute, for combinations such as a causal-only
    variant configured with bidirectional attention, a ``max_length`` that is not a multiple of
    ``block_size``, or a missing time conditioning. Also a :class:`ValueError` for callers that
    predate the hierarchy.
    """
