"""MTP eval metric: greedy accuracy of every head on its own offset."""

from __future__ import annotations

from hale_llm.core.registry import register_metric
from hale_llm.metrics.base import Metric, MetricContext


@register_metric("head_accuracy", variants=("mtp",))
class MTPHeadAccuracy(Metric):
    """``head{k}_accuracy``: how often head ``k`` predicts token ``i + k + 1`` exactly.

    Head accuracies bound self-speculative decoding: the expected accepted drafts per forward
    grow with how well heads ``1..K-1`` agree with the main head.
    """

    metric_name = "head_accuracy"

    def __init__(self) -> None:
        self.correct: dict[int, float] = {}
        self.total: dict[int, float] = {}

    def reset(self) -> None:
        """Clear the running sums (called before every evaluation pass)."""
        self.correct, self.total = {}, {}

    def update(self, ctx: MetricContext) -> None:
        """Add each head's exact-match hits on its own offset for this batch."""
        labels, pad = ctx.batch["labels"], ctx.batch["pad_token_id"]
        for k, logits in enumerate(ctx.model(ctx.batch)):
            shift = k + 1
            if labels.size(1) <= shift:
                break
            target = labels[:, shift:]
            valid = target.ne(pad)
            pred = logits[:, :-shift].argmax(dim=-1)
            self.correct[k] = self.correct.get(k, 0.0) + float(((pred == target) & valid).sum())
            self.total[k] = self.total.get(k, 0.0) + float(valid.sum())

    def compute(self) -> dict[str, float]:
        """``{'head{k}_accuracy': value}`` for every head that saw a valid target."""
        return {
            f"head{k}_accuracy": self.correct[k] / max(self.total[k], 1.0)
            for k in sorted(self.total)
        }
