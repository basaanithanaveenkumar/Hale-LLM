"""
Distinct-N diversity metric for generated text.

Li et al. (2016) "A Diversity-Promoting Objective Function for Neural
Conversation Models" — https://arxiv.org/abs/1510.03055

Why Distinct-N?
---------------
Standard metrics like BLEU measure accuracy against a reference.  Distinct-N
measures *diversity* across the generated outputs themselves.

Distinct-N = (unique n-grams in all generated text) / (total n-grams)

A model that always generates the same output will score Distinct-1 ≈ 0.
A model with rich vocabulary and varied syntax will score near 1.

Low Distinct-N is a sign of mode collapse — the model has learned to produce
"safe", average responses rather than varied, informative ones.
"""

from __future__ import annotations

from collections import Counter

from hale_llm.core.registry import register_metric
from hale_llm.metrics.base import Metric, MetricContext


@register_metric("distinct_n")
class DistinctNMetric(Metric):
    """Distinct-1 and Distinct-2 diversity metrics.

    Reads ``ctx.batch["generated_text"]`` (list[str]).

    Results
    -------
    distinct_1  : fraction of unique unigrams  in all outputs (higher = more diverse).
    distinct_2  : fraction of unique bigrams   in all outputs.
    vocab_size  : total unique unigrams seen.
    """

    metric_name = "distinct_n"

    def reset(self) -> None:
        # Accumulate n-gram counters across the whole corpus
        """Clear the running sums (called before every evaluation pass)."""
        self._unigrams: Counter = Counter()
        self._bigrams: Counter = Counter()

    def update(self, ctx: MetricContext) -> None:
        """Accumulate n-gram counts from generated text.

        Expects:
            ctx.batch["generated_text"]: list[str]
        """
        for text in ctx.batch.get("generated_text", []):
            toks = text.split()
            self._unigrams.update(toks)
            self._bigrams.update(zip(toks, toks[1:]))

    def compute(self) -> dict[str, float]:
        """Return Distinct-1, Distinct-2, and vocabulary size.

        Returns:
            distinct_1:  unique unigrams / total unigrams.
            distinct_2:  unique bigrams  / total bigrams.
            vocab_size:  number of unique tokens seen.
        """
        total_1 = sum(self._unigrams.values())
        total_2 = sum(self._bigrams.values())
        return {
            "distinct_1": len(self._unigrams) / max(total_1, 1),
            "distinct_2": len(self._bigrams) / max(total_2, 1),
            "vocab_size": float(len(self._unigrams)),
        }
