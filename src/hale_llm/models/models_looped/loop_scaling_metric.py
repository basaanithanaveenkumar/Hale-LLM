"""Next-token loss at several loop budgets (test-time compute scaling)."""

from __future__ import annotations

from hale_llm.core.registry import register_metric
from hale_llm.metrics.base import Metric, MetricContext
from hale_llm.models.models_looped.next_token_nll import next_token_nll


@register_metric("loop_scaling", variants=("looped",))
class LoopScalingMetric(Metric):
    """Next-token loss at several loop budgets: does thinking longer help?

    Evaluates ``loss@T`` for ``T`` in ``{1, T/2, T, 2T}`` (``T = loop.n_loops``; budgets above
    ``T`` only with the sinusoidal depth embedding, which is defined for any step). A model
    trained with ``min_loops`` / ``bptt_loops`` should improve monotonically up to ``T`` and
    hold (or keep improving) past it; a fixed-``T`` model typically degrades off its ``T``.
    """

    metric_name = "loop_scaling"

    def __init__(self) -> None:
        self.sums: dict[int, float] = {}
        self.n = 0

    def reset(self) -> None:
        self.sums = {}
        self.n = 0

    @staticmethod
    def budgets(model) -> list[int]:
        spec = model.loop
        out = {1, max(1, spec.n_loops // 2), spec.n_loops}
        if spec.depth_embedding != "learned":
            out.add(2 * spec.n_loops)
        return sorted(out)

    def update(self, ctx: MetricContext) -> None:
        model = ctx.model
        for budget in self.budgets(model):
            loss = next_token_nll(model, ctx.batch, model.forward_with_loops(ctx.batch, budget))
            self.sums[budget] = self.sums.get(budget, 0.0) + loss
        self.n += 1

    def compute(self) -> dict[str, float]:
        if not self.n:
            return {}
        return {f"loss@T{budget}": total / self.n for budget, total in sorted(self.sums.items())}
