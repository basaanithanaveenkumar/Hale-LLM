"""Looped-transformer compute metric: how many passes the shared block actually ran."""

from __future__ import annotations

from hale_core.nn.stacks import loop_stats

from hale_llm.core.registry import register_metric
from hale_llm.metrics.base import Metric, MetricContext


@register_metric("loop_depth")
class LoopDepthMetric(Metric):
    """Mean loop statistics over the eval batches (``arch: looped`` only; empty otherwise).

    Reports ``loops`` (passes run; below ``n_loops`` after ACT / fixed-point early exit),
    ``mean_updates`` (ACT: average passes per token) and ``ponder_cost`` (ACT: ``N + R``),
    read from the backbone after the variant's loss ran its forward.
    """

    metric_name = "loop_depth"

    def __init__(self) -> None:
        self.sums: dict[str, float] = {}
        self.n = 0

    def reset(self) -> None:
        """Clear the running sums (called before every evaluation pass)."""
        self.sums = {}
        self.n = 0

    def update(self, ctx: MetricContext) -> None:
        """Add the backbone's loop statistics from the forward the loss just ran."""
        stats = loop_stats(ctx.model)
        if not stats:
            return
        for key, value in stats.items():
            name = key.split("/", 1)[1] if key.startswith("loop/") else key.replace("/", "_")
            self.sums[name] = self.sums.get(name, 0.0) + value
        self.n += 1

    def compute(self) -> dict[str, float]:
        """Mean of each loop statistic over the batches (empty for non-looped models)."""
        return {name: total / self.n for name, total in self.sums.items()} if self.n else {}
