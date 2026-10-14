"""
hale_llm.metrics.generation — Text generation quality metrics.

All metrics use the ``MetricContext`` + ``@register_metric`` pattern from
``hale_llm.metrics.base`` and ``hale_llm.core.registry``.

Metrics
-------
BLEUMetric        — Corpus BLEU-1/2/3/4 + geometric mean (Papineni 2002)
ROUGEMetric       — ROUGE-1, ROUGE-2, ROUGE-L F1 (Lin 2004)
DistinctNMetric   — Distinct-1 and Distinct-2 diversity (Li 2016)
SelfBLEUMetric    — Intra-generation diversity via self-scoring (Zhu 2018)
RepetitionMetric  — Per-sentence n-gram repetition rate (Welleck 2020)
"""

from hale_llm.metrics.generation.bleu import BLEUMetric
from hale_llm.metrics.generation.distinct import DistinctNMetric
from hale_llm.metrics.generation.repetition import RepetitionMetric
from hale_llm.metrics.generation.rouge import ROUGEMetric
from hale_llm.metrics.generation.self_bleu import SelfBLEUMetric

__all__ = [
    "BLEUMetric",
    "ROUGEMetric",
    "DistinctNMetric",
    "SelfBLEUMetric",
    "RepetitionMetric",
]
