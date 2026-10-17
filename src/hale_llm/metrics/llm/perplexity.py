"""Perplexity derived from the mean loss (``perplexity``)."""

import math

from hale_llm.core.registry import register_metric
from hale_llm.metrics.base import Metric
from hale_llm.metrics.llm.common import MeanLossMixin


@register_metric("perplexity")
class PerplexityMetric(MeanLossMixin, Metric):
    """exp(mean batch loss). Comparable across variants only when the loss is token NLL."""

    def compute(self) -> dict[str, float]:
        """``{'perplexity': exp(mean_loss)}`` (``inf`` on overflow)."""
        mean = self.mean_loss()
        try:
            ppl = math.exp(mean)
        except OverflowError:
            ppl = float("inf")
        return {"perplexity": ppl}
