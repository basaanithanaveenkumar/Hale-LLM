"""Generation for the looped model: the plain token-by-token sampler (each step runs the configured number of loops)."""

from hale_llm.models.models_autoregressive.sample import sample_autoregressive
from hale_llm.models.models_looped.variants import ALL_VARIANTS
from hale_llm.registry import register_sampler

for _name in ALL_VARIANTS:
    register_sampler(_name)(sample_autoregressive)
