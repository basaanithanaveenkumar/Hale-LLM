"""Base exception of the online RL package."""

from __future__ import annotations


class OnlineRLError(RuntimeError):
    """Raised for invalid online-RL configurations or states (bad group size, unknown reward, …)."""
