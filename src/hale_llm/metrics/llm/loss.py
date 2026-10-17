"""Mean evaluation loss (``loss``)."""

from hale_llm.core.registry import register_metric
from hale_llm.metrics.base import Metric
from hale_llm.metrics.llm.common import MeanLossMixin


@register_metric("loss")
class LossMetric(MeanLossMixin, Metric):
    """Mean of the variant's own training loss over the evaluation batches."""
    def compute(self) -> dict[str, float]:
        """``{'loss': mean batch loss}``."""
        return {"loss": self.mean_loss()}
