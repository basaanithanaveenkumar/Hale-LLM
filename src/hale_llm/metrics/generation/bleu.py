"""
BLEU metric (corpus-level) for text generation evaluation.

BLEU (Bilingual Evaluation Understudy, Papineni et al. 2002) measures how much
the n-grams of a generated hypothesis overlap with those of a reference.

Key formula components
----------------------
1. N-gram Precision (modified clip count):
   For each n-gram in the hypothesis, count how many times it appears in
   the reference, but clip to the maximum count in any single reference.
   This prevents gaming the metric by repeating common n-grams.

       p_n = Σ_hypothesis  clip_count(n-gram) / Σ_hypothesis  count(n-gram)

2. Brevity Penalty (BP):
   Penalises hypotheses shorter than their references, since short outputs
   trivially achieve high precision by only generating certain n-grams.

       BP = 1                         if c > r
            exp(1 - r/c)              otherwise

   where c = hypothesis length, r = closest reference length.

3. Final BLEU-N:
   BLEU-N = BP * exp( (1/N) * Σ_{n=1..N} log(p_n) )

   BLEU-AVG (used here) = geometric mean of BLEU-1 through BLEU-4.
"""

from __future__ import annotations

import math
from collections import Counter

from hale_llm.core.registry import register_metric
from hale_llm.metrics.base import Metric, MetricContext


def _get_ngrams(tokens: list[str], n: int) -> Counter:
    """Extract all n-grams from a token list as a Counter."""
    return Counter(tuple(tokens[i: i + n]) for i in range(len(tokens) - n + 1))


def _clip_count(hyp_ngrams: Counter, ref_ngrams: Counter) -> int:
    """Clipped n-gram count: min(hyp_count, ref_count) summed over all n-grams."""
    return sum(min(count, ref_ngrams[ngram]) for ngram, count in hyp_ngrams.items())


@register_metric("bleu")
class BLEUMetric(Metric):
    """Corpus-level BLEU-1/2/3/4 and geometric mean BLEU-AVG.

    Reads ``ctx.batch["generated_text"]`` (list[str]) and
    ``ctx.batch["reference_text"]`` (list[str]).

    Results
    -------
    bleu_1, bleu_2, bleu_3, bleu_4 : Sentence n-gram BLEU scores (0–100).
    bleu_avg                        : Geometric mean of BLEU 1–4.
    """

    metric_name = "bleu"

    def reset(self) -> None:
        # Accumulate corpus-level counts across all batches
        self._clip_counts: list[int] = [0, 0, 0, 0]    # for n=1..4
        self._total_counts: list[int] = [0, 0, 0, 0]
        self._hyp_len: int = 0   # total hypothesis token count
        self._ref_len: int = 0   # total reference token count

    def update(self, ctx: MetricContext) -> None:
        """Accumulate BLEU statistics from one batch.

        Expects:
            ctx.batch["generated_text"]: list[str] — model outputs.
            ctx.batch["reference_text"]: list[str] — gold references.
        """
        hyps: list[str] = ctx.batch.get("generated_text", [])
        refs: list[str] = ctx.batch.get("reference_text", [])

        for hyp, ref in zip(hyps, refs):
            hyp_toks = hyp.split()
            ref_toks = ref.split()

            self._hyp_len += len(hyp_toks)
            self._ref_len += len(ref_toks)

            for n in range(1, 5):
                hyp_ngrams = _get_ngrams(hyp_toks, n)
                ref_ngrams = _get_ngrams(ref_toks, n)
                self._clip_counts[n - 1] += _clip_count(hyp_ngrams, ref_ngrams)
                self._total_counts[n - 1] += max(len(hyp_toks) - n + 1, 0)

    def compute(self) -> dict[str, float]:
        """Compute corpus-level BLEU scores.

        Returns:
            bleu_1, bleu_2, bleu_3, bleu_4 (0–100 scale).
            bleu_avg: geometric mean of bleu_1..bleu_4.
        """
        # Brevity penalty: penalise hypotheses shorter than references
        c, r = self._hyp_len, self._ref_len
        if c == 0:
            return {f"bleu_{n}": 0.0 for n in range(1, 5)} | {"bleu_avg": 0.0}

        bp = 1.0 if c >= r else math.exp(1.0 - r / c)

        log_avg = 0.0
        results: dict[str, float] = {}
        for n in range(1, 5):
            denom = max(self._total_counts[n - 1], 1)
            prec = self._clip_counts[n - 1] / denom
            # Use smoothed precision (add 1 to avoid log(0) for high n)
            prec = max(prec, 1e-12)
            bleu_n = 100.0 * bp * math.exp(math.log(prec))
            results[f"bleu_{n}"] = bleu_n
            log_avg += math.log(prec)

        results["bleu_avg"] = 100.0 * bp * math.exp(log_avg / 4)
        return results
