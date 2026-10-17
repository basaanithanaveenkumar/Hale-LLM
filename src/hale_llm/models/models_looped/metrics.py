"""Looped-LM metrics: test-time compute scaling and per-token pondering.

Importing this module registers both metrics (``loop_scaling``, ``ponder_histogram``).
"""

from hale_llm.models.models_looped.loop_scaling_metric import LoopScalingMetric
from hale_llm.models.models_looped.ponder_histogram_metric import PonderHistogramMetric

__all__ = ["LoopScalingMetric", "PonderHistogramMetric"]
