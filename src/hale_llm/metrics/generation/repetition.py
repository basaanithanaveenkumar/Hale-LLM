"""
Repetition metric for generated text.

Welleck et al. (2020) "Neural Text Generation With Unlikelihood Training"
— https://arxiv.org/abs/1908.04319

What is Repetition-N?
---------------------
rep_n = 1 - (unique n-grams in text) / (total n-grams in text)

High rep_n → many repeated n-grams → the model is stuck in loops.
Low  rep_n → varied n-gram usage → no repetition.

This is measured *within* each individual generation (not across the corpus),
making it complementary to Distinct-N which measures diversity *across*
all generated outputs.

Example
-------
"the cat sat on the mat the cat sat"  (length 9)
bigrams: {("the","cat"):2, ("cat","sat"):2, ("sat","on"):1, ("on","the"):1,
           ("the","mat"):1, ("mat","the"):1}
total bigrams = 8, unique = 6  →  rep_2 = 1 - 6/8 = 0.25
"""

from __future__ import annotations

from hale_llm.core.registry import register_metric
from hale_llm.metrics.base import Metric, MetricContext


def _rep_n(tokens: list[str], n: int) -> float:
    """Per-sentence repetition rate for n-grams."""
    if len(tokens) < n:
        return 0.0
    ngrams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
    total = len(ngrams)
    unique = len(set(ngrams))
    return 1.0 - unique / total


@register_metric("repetition")
class RepetitionMetric(Metric):
    """Per-sentence n-gram repetition rates (rep_1, rep_2, rep_3).

    Reads ``ctx.batch["generated_text"]`` (list[str]).

    Results
    -------
    rep_1, rep_2, rep_3 : mean repetition rate for unigrams, bigrams, trigrams.
                          Values in [0, 1]; lower = less repetitive.
    """

    metric_name = "repetition"

    def reset(self) -> None:
        """Clear the running sums (called before every evaluation pass)."""
        self._rep_scores: dict[int, list[float]] = {1: [], 2: [], 3: []}

    def update(self, ctx: MetricContext) -> None:
        """Accumulate repetition rates from one batch.

        Expects:
            ctx.batch["generated_text"]: list[str]
        """
        for text in ctx.batch.get("generated_text", []):
            toks = text.split()
            for n in (1, 2, 3):
                self._rep_scores[n].append(_rep_n(toks, n))

    def compute(self) -> dict[str, float]:
        """Return mean repetition rates.

        Returns:
            rep_1, rep_2, rep_3: mean repetition rate per n (0 = no repetition).
        """
        results: dict[str, float] = {}
        for n, scores in self._rep_scores.items():
            results[f"rep_{n}"] = sum(scores) / len(scores) if scores else 0.0
        return results
