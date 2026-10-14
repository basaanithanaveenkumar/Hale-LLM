"""
hale_llm.metrics — All evaluation metrics for HaleDiffusionLLM.

Subpackages
-----------
llm         — Language model metrics: loss, perplexity, token accuracy
generation  — Text generation quality: BLEU, ROUGE, Distinct-N, Self-BLEU, Repetition
diffusion   — Diffusion process metrics: NFE, VLB, acceptance rate
"""

from hale_llm.metrics.base import Metric, MetricContext

# LLM metrics
from hale_llm.metrics.llm import (
    BitsPerTokenMetric,
    LossMetric,
    PerplexityMetric,
    TokenAccuracyMetric,
    TopKAccuracyMetric,
)

# Generation quality metrics
from hale_llm.metrics.generation import (
    BLEUMetric,
    DistinctNMetric,
    RepetitionMetric,
    ROUGEMetric,
    SelfBLEUMetric,
)

# Diffusion process metrics
from hale_llm.metrics.diffusion import (
    AcceptanceRateMetric,
    NFEMetric,
    VLBMetric,
)

__all__ = [
    # Base
    "Metric",
    "MetricContext",
    # LLM
    "LossMetric",
    "PerplexityMetric",
    "BitsPerTokenMetric",
    "TokenAccuracyMetric",
    "TopKAccuracyMetric",
    # Generation
    "BLEUMetric",
    "ROUGEMetric",
    "DistinctNMetric",
    "SelfBLEUMetric",
    "RepetitionMetric",
    # Diffusion
    "NFEMetric",
    "VLBMetric",
    "AcceptanceRateMetric",
]
