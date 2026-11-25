from __future__ import annotations

from hale_llm.registry import register_metric
from hale_llm.metrics.base import Metric, MetricContext
from hale_llm.metrics.llm.variants import AUTOREGRESSIVE_VARIANTS


@register_metric("topk_accuracy", variants=AUTOREGRESSIVE_VARIANTS)
class TopKAccuracyMetric(Metric):
    """Fraction of non-ignored labels whose true id is in the top-k logits."""

    def __init__(self, k: int = 5) -> None:
        self.k = k
        self.correct = 0.0
        self.total = 0.0

    def reset(self) -> None:
        self.correct = 0.0
        self.total = 0.0

    def update(self, ctx: MetricContext) -> None:
        labels = ctx.batch.get("labels")
        if labels is None:
            return
        logits = ctx.model(ctx.batch)
        if isinstance(logits, (list, tuple)):
            logits = logits[0]
        pad_id = ctx.batch.get("pad_token_id")
        valid = labels.ne(-100)
        if pad_id is not None:
            valid = valid & labels.ne(pad_id)
        k = min(self.k, logits.size(-1))
        topk = logits.topk(k, dim=-1).indices
        hit = (topk == labels.unsqueeze(-1)).any(dim=-1)
        n = float(valid.float().sum().item())
        if n == 0:
            return
        self.correct += float((hit & valid).float().sum().item())
        self.total += n

    def compute(self) -> dict[str, float]:
        return {"topk_accuracy": self.correct / max(self.total, 1.0)}
