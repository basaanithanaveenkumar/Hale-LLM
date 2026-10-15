"""Looped transformer LM: one weight-tied block iterated T times (docs/looped.md).

model.py    @register_variant("looped")  LoopedLM over hale_core's LoopedBackbone
loss.py     @register_loss("looped")     next-token NLL (the trainer adds the ACT ponder cost)
sample.py   @register_sampler("looped")  autoregressive decoding under a test-time loop budget
metrics.py  @register_metric(..., variants=("looped",))  loss vs. loop count, per-token ponder
"""

from hale_llm.models.models_looped import loss as _loss  # noqa: F401
from hale_llm.models.models_looped import metrics as _metrics  # noqa: F401
from hale_llm.models.models_looped import sample as _sample  # noqa: F401
from hale_llm.models.models_looped.model import LoopedLM, collate_looped

__all__ = ["LoopedLM", "collate_looped"]
