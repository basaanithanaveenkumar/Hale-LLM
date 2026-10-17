"""Backward-compatible home of the metric base types.

:class:`Metric` lives in :mod:`hale_llm.metrics.metric` and :class:`MetricContext` in
:mod:`hale_llm.metrics.metric_context`; both are re-exported here for existing imports.
"""

from hale_llm.metrics.metric import Metric
from hale_llm.metrics.metric_context import MetricContext

__all__ = ["Metric", "MetricContext"]
