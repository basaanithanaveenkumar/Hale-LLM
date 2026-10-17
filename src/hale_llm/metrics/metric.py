"""Abstract base class of every evaluation metric."""

from __future__ import annotations

from abc import ABC, abstractmethod

from hale_llm.metrics.metric_context import MetricContext


class Metric(ABC):
    """Accumulate over an eval loader, then `compute()` a dict of scalar names."""

    metric_name: str = ""

    def reset(self) -> None:
        return

    @abstractmethod
    def update(self, ctx: MetricContext) -> None:
        raise NotImplementedError

    @abstractmethod
    def compute(self) -> dict[str, float]:
        raise NotImplementedError
