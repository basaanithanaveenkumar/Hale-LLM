"""Abstract base class of every evaluation metric."""

from __future__ import annotations

from abc import ABC, abstractmethod

from hale_llm.metrics.metric_context import MetricContext


class Metric(ABC):
    """Accumulate over an eval loader, then `compute()` a dict of scalar names."""

    metric_name: str = ""

    def reset(self) -> None:
        """Clear accumulated state before an evaluation pass (default: nothing to clear)."""
        return

    @abstractmethod
    def update(self, ctx: MetricContext) -> None:
        """Accumulate statistics from one batch (``ctx`` holds model, batch, loss and variant)."""
        raise NotImplementedError

    @abstractmethod
    def compute(self) -> dict[str, float]:
        """Return ``{name: value}`` scalars for everything accumulated since :meth:`reset`."""
        raise NotImplementedError
