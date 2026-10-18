"""Shared running mean of the batch loss for loss-derived metrics."""

from hale_llm.metrics.base import MetricContext


class MeanLossMixin:
    """Mixin keeping a running mean of ``ctx.loss`` (used by loss, perplexity and bits per
    token).
    """

    def __init__(self) -> None:
        self.total = 0.0
        self.n = 0

    def reset(self) -> None:
        """Clear the running sums (called before every evaluation pass)."""
        self.total = 0.0
        self.n = 0

    def update(self, ctx: MetricContext) -> None:
        """Add this batch's scalar loss to the running sum."""
        self.total += float(ctx.loss.item())
        self.n += 1

    def mean_loss(self) -> float:
        """Mean batch loss so far (0 if no batch was seen)."""
        return self.total / max(self.n, 1)
