"""Base exception of the online RL package."""

from __future__ import annotations

from hale_llm.exceptions import HaleLLMError


class OnlineRLError(HaleLLMError, RuntimeError):
    """Invalid online-RL configuration or state (bad group size, unknown reward, …).

    Part of the :mod:`hale_llm.exceptions` hierarchy; also a :class:`RuntimeError` (its original
    base) so existing handlers keep working.
    """
