"""
ROUGE metrics for text generation evaluation.

ROUGE (Recall-Oriented Understudy for Gisting Evaluation, Lin 2004) measures
overlap between generated and reference text.  Unlike BLEU (which is precision-
oriented), ROUGE emphasises recall.  All scores here are F1 = harmonic mean of
precision and recall.

ROUGE-1 and ROUGE-2
-------------------
Standard n-gram overlap.  ROUGE-1 captures content word overlap; ROUGE-2
additionally checks for phrase continuity.

    precision_n = |overlap| / |hypothesis n-grams|
    recall_n    = |overlap| / |reference n-grams|
    F1_n        = 2 * P * R / (P + R)

ROUGE-L (Longest Common Subsequence)
--------------------------------------
Measures the longest in-order match between hypothesis and reference without
requiring consecutive tokens — more lenient than exact phrase matching.

LCS algorithm (dynamic programming):
    L[i][j] = 0                          if i==0 or j==0
             = L[i-1][j-1] + 1           if hyp[i] == ref[j]
             = max(L[i-1][j], L[i][j-1]) otherwise

    lcs_len = L[len(hyp)][len(ref)]

ROUGE-L F1 = 2 * lcs_len / (len(hyp) + len(ref))
"""

from __future__ import annotations

from collections import Counter

from hale_llm.core.registry import register_metric
from hale_llm.metrics.base import Metric, MetricContext


def _ngram_overlap_f1(hyp_toks: list[str], ref_toks: list[str], n: int) -> float:
    """Compute n-gram overlap F1 between hypothesis and reference."""
    hyp_ng = Counter(tuple(hyp_toks[i : i + n]) for i in range(max(len(hyp_toks) - n + 1, 0)))
    ref_ng = Counter(tuple(ref_toks[i : i + n]) for i in range(max(len(ref_toks) - n + 1, 0)))

    overlap = sum(min(hyp_ng[g], ref_ng[g]) for g in hyp_ng)
    hyp_total = sum(hyp_ng.values())
    ref_total = sum(ref_ng.values())

    if hyp_total == 0 or ref_total == 0:
        return 0.0

    prec = overlap / hyp_total
    rec = overlap / ref_total
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


def _lcs_length(a: list[str], b: list[str]) -> int:
    """Longest Common Subsequence length via O(|a|*|b|) DP.

    Uses a rolling two-row array to keep memory at O(min(|a|,|b|)).
    """
    # Ensure 'a' is the shorter sequence (minimises memory)
    if len(a) > len(b):
        a, b = b, a
    n = len(a)
    prev = [0] * (n + 1)
    curr = [0] * (n + 1)
    for bj in b:
        for i, ai in enumerate(a, start=1):
            if ai == bj:
                curr[i] = prev[i - 1] + 1
            else:
                curr[i] = max(curr[i - 1], prev[i])
        prev, curr = curr, [0] * (n + 1)
    return prev[n]


def _rouge_l_f1(hyp_toks: list[str], ref_toks: list[str]) -> float:
    """ROUGE-L F1 based on Longest Common Subsequence."""
    if not hyp_toks or not ref_toks:
        return 0.0
    lcs = _lcs_length(hyp_toks, ref_toks)
    prec = lcs / len(hyp_toks)
    rec = lcs / len(ref_toks)
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


@register_metric("rouge")
class ROUGEMetric(Metric):
    """Corpus-level ROUGE-1, ROUGE-2, and ROUGE-L (F1).

    Reads ``ctx.batch["generated_text"]`` and ``ctx.batch["reference_text"]``.

    Results
    -------
    rouge_1, rouge_2, rouge_l : F1 scores in the range [0, 100].
    """

    metric_name = "rouge"

    def reset(self) -> None:
        """Clear the running sums (called before every evaluation pass)."""
        self._scores: dict[str, list[float]] = {
            "rouge_1": [],
            "rouge_2": [],
            "rouge_l": [],
        }

    def update(self, ctx: MetricContext) -> None:
        """Accumulate ROUGE scores for one batch.

        Expects:
            ctx.batch["generated_text"]: list[str]
            ctx.batch["reference_text"]: list[str]
        """
        hyps: list[str] = ctx.batch.get("generated_text", [])
        refs: list[str] = ctx.batch.get("reference_text", [])

        for hyp, ref in zip(hyps, refs):
            hyp_toks = hyp.split()
            ref_toks = ref.split()
            self._scores["rouge_1"].append(_ngram_overlap_f1(hyp_toks, ref_toks, 1))
            self._scores["rouge_2"].append(_ngram_overlap_f1(hyp_toks, ref_toks, 2))
            self._scores["rouge_l"].append(_rouge_l_f1(hyp_toks, ref_toks))

    def compute(self) -> dict[str, float]:
        """Return mean ROUGE scores scaled to [0, 100].

        Returns:
            rouge_1, rouge_2, rouge_l: corpus-mean F1 (%).
        """
        results: dict[str, float] = {}
        for key, vals in self._scores.items():
            results[key] = 100.0 * (sum(vals) / len(vals)) if vals else 0.0
        return results
