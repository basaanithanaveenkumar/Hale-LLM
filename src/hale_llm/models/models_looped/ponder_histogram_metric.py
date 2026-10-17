"""Histogram of per-token pondering (loops used) under ACT halting."""

from __future__ import annotations

import torch

from hale_llm.core.registry import register_metric
from hale_llm.metrics.base import Metric, MetricContext


@register_metric("ponder_histogram", variants=("looped",))
class PonderHistogramMetric(Metric):
    """ACT only: how the passes per token are distributed over the eval tokens.

    Reports ``ponder_mean``, ``ponder_min`` / ``ponder_max`` and ``ponder_frac@k`` (share of real
    tokens that halted after exactly ``k`` passes). Empty for models without ACT.
    """

    metric_name = "ponder_histogram"

    def __init__(self) -> None:
        self.counts: dict[int, float] = {}
        self.total = 0.0

    def reset(self) -> None:
        self.counts = {}
        self.total = 0.0

    def update(self, ctx: MetricContext) -> None:
        model = ctx.model
        if model.loop.halting != "act":
            return
        model(ctx.batch)
        n_updates = model.last_loop_stats.token_updates
        mask = ctx.batch.get("attention_mask")
        valid = mask.bool() if mask is not None else torch.ones_like(n_updates, dtype=torch.bool)
        values, counts = n_updates[valid].round().long().unique(return_counts=True)
        for value, count in zip(values.tolist(), counts.tolist(), strict=True):
            self.counts[value] = self.counts.get(value, 0.0) + count
        self.total += float(valid.sum())

    def compute(self) -> dict[str, float]:
        if not self.total:
            return {}
        out = {f"ponder_frac@{k}": c / self.total for k, c in sorted(self.counts.items())}
        out["ponder_mean"] = sum(k * c for k, c in self.counts.items()) / self.total
        out["ponder_min"] = float(min(self.counts))
        out["ponder_max"] = float(max(self.counts))
        return out
