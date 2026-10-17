"""Block flow matching eval metric: accuracy on the masked positions of a corrupted batch."""

from __future__ import annotations

from hale_llm.core.registry import register_metric
from hale_llm.metrics.base import Metric, MetricContext
from hale_llm.models.models_block_flow_matching.path import corrupt_blocks


@register_metric("masked_accuracy", variants=("block_flow_matching",))
class BlockFlowMatchingMaskedAccuracy(Metric):
    """Greedy ``x_1`` accuracy at masked positions, one random ``t`` per block."""

    def __init__(self) -> None:
        self.correct = 0.0
        self.total = 0.0

    def reset(self) -> None:
        """Clear the running sums (called before every evaluation pass)."""
        self.correct = 0.0
        self.total = 0.0

    def update(self, ctx: MetricContext) -> None:
        """Accumulate correct predictions on masked positions from one evaluation batch."""
        model, x1 = ctx.model, ctx.batch["input_ids"]
        x_t, is_masked, t = corrupt_blocks(
            x1, model.block_size, ctx.batch["mask_token_id"], model.path
        )
        logits = model({**ctx.batch, "x_clean": x1, "x_t": x_t, "t": t})
        valid = is_masked
        if ctx.batch.get("attention_mask") is not None:
            valid = valid & ctx.batch["attention_mask"].bool()
        n = float(valid.float().sum())
        if n == 0:
            return
        self.correct += float(((logits.argmax(dim=-1) == x1) & valid).float().sum())
        self.total += n

    def compute(self) -> dict[str, float]:
        """``{'masked_accuracy': correct / total}`` over the batches seen (0 if none)."""
        return {"masked_accuracy": self.correct / max(self.total, 1.0)}
