"""The looped model trains on the plain next-token loss (plus the MoE balancing loss when its blocks are experts)."""

from hale_llm.models.models_autoregressive.loss import autoregressive_loss
from hale_llm.models.models_looped.variants import ALL_VARIANTS
from hale_llm.registry import register_loss

for _name in ALL_VARIANTS:
    register_loss(_name)(autoregressive_loss)
