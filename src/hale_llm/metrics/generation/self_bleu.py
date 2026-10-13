"""
Self-BLEU — intra-diversity metric for text generation.

Zhu et al. (2018) "Texygen: A Benchmarking Platform for Text Generation Models"
— https://arxiv.org/abs/1802.01886

What is Self-BLEU?
------------------
Self-BLEU treats each generated sentence as the hypothesis and all OTHER
generated sentences as references.  It then computes standard BLEU.

    self_bleu = mean_over_sentences( BLEU(sentence, all_others) )

Interpretation
--------------
  High self-BLEU  → generated sentences are very similar to each other
                    (mode collapse, lack of diversity)
  Low  self-BLEU  → generated sentences are varied and lexically diverse

Unlike Distinct-N which measures token-level diversity, Self-BLEU captures
phrase-level and structural diversity.

Note: Self-BLEU is computed in O(N²) in the number of sentences.  For large
corpora, set ``max_sentences`` to subsample.
"""

from __future__ import annotations

import math
from collections import Counter

from hale_llm.core.registry import register_metric
from hale_llm.metrics.base import Metric, MetricContext


def _bleu_1(hyp_toks: list[str], ref_toks_list: list[list[str]]) -> float:
    """Compute simple unigram BLEU of hyp against a set of references.

    Uses clipped precision and brevity penalty.
    """
    hyp_len = len(hyp_toks)
    if hyp_len == 0:
        return 0.0

    hyp_counts = Counter(hyp_toks)
    # Max ref count per n-gram across all references (multi-reference clipping)
    max_ref_counts: Counter = Counter()
    for ref_toks in ref_toks_list:
        ref_counts = Counter(ref_toks)
        for tok in hyp_counts:
            max_ref_counts[tok] = max(max_ref_counts[tok], ref_counts[tok])

    clipped = sum(min(hyp_counts[t], max_ref_counts[t]) for t in hyp_counts)
    prec = clipped / hyp_len if hyp_len > 0 else 0.0

    # Brevity penalty against the closest (in length) reference
    ref_lens = [len(r) for r in ref_toks_list]
    closest_len = min(ref_lens, key=lambda rl: (abs(rl - hyp_len), rl))
    bp = 1.0 if hyp_len >= closest_len else math.exp(1.0 - closest_len / hyp_len)

    return bp * prec


@register_metric("self_bleu")
class SelfBLEUMetric(Metric):
    """Self-BLEU: measures intra-generation diversity.

    Reads only ``ctx.batch["generated_text"]`` (no references needed).

    Results
    -------
    self_bleu: Mean BLEU of each sentence against all others (0–100 scale).
               Lower values indicate more diverse generation.
    """

    metric_name = "self_bleu"

    def __init__(self, max_sentences: int = 1000) -> None:
        """
        Args:
            max_sentences: Cap on accumulated sentences to bound O(N²) cost.
        """
        self.max_sentences = max_sentences
        self.reset()

    def reset(self) -> None:
        self._sentences: list[list[str]] = []

    def update(self, ctx: MetricContext) -> None:
        """Accumulate generated sentences.

        Expects:
            ctx.batch["generated_text"]: list[str]
        """
        for text in ctx.batch.get("generated_text", []):
            if len(self._sentences) < self.max_sentences:
                self._sentences.append(text.split())

    def compute(self) -> dict[str, float]:
        """Compute mean Self-BLEU over all accumulated sentences.

        Returns:
            self_bleu: Mean BLEU-1 of each sentence vs. all others (0–100).
        """
        sents = self._sentences
        if len(sents) < 2:
            return {"self_bleu": 0.0}

        scores: list[float] = []
        for i, hyp in enumerate(sents):
            refs = [s for j, s in enumerate(sents) if j != i]
            scores.append(_bleu_1(hyp, refs))

        return {"self_bleu": 100.0 * (sum(scores) / len(scores))}
