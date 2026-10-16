"""Multi-token prediction: K heads on one trunk, self-speculative decoding.

model.py    @register_variant("mtp")  MultiTokenPredictionLM (+ MTPHead, collate)
loss.py     @register_loss("mtp")     mean of the per-head shifted NLLs
sample.py   @register_sampler("mtp")  draft with heads 1..K-1, verify with head 0
metrics.py  head_accuracy
"""

from hale_llm.models.models_mtp import metrics as _metrics  # noqa: F401
from hale_llm.models.models_mtp.loss import head_losses, mtp_loss
from hale_llm.models.models_mtp.model import MTPHead, MultiTokenPredictionLM, collate_mtp
from hale_llm.models.models_mtp.sample import SpeculativeStep, sample_mtp

__all__ = [
    "MTPHead",
    "MultiTokenPredictionLM",
    "SpeculativeStep",
    "collate_mtp",
    "head_losses",
    "mtp_loss",
    "sample_mtp",
]
