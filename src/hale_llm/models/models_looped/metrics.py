"""Looped-LM metrics: test-time compute scaling and per-token pondering."""

from __future__ import annotations

import torch

from hale_llm.core.registry import register_metric
from hale_llm.losses import model_token_nll
from hale_llm.metrics.base import Metric, MetricContext


def _nll(model, batch: dict, logits: torch.Tensor) -> float:
    return float(
        model_token_nll(
            model, logits[:, :-1, :], batch["labels"][:, :-1], ignore_index=-100, reduction="mean"
        )
    )


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
            loss = _nll(model, ctx.batch, model.forward_with_loops(ctx.batch, budget))
            self.sums[budget] = self.sums.get(budget, 0.0) + loss
        self.n += 1

    def compute(self) -> dict[str, float]:
        if not self.n:
            return {}
        return {f"loss@T{budget}": total / self.n for budget, total in sorted(self.sums.items())}


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
