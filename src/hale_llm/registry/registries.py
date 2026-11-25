"""The registries every plugin decorator writes to."""

from __future__ import annotations

from hale_llm.registry.named_registry import NamedRegistry
from hale_llm.registry.variant_registry import VariantRegistry

#: Model variants (``variant:`` in YAML), e.g. ``autoregressive``.
MODELS_REGISTRY = NamedRegistry("model")
#: Training losses, keyed by variant.
LOSSES_REGISTRY = NamedRegistry("loss")
#: Text samplers, keyed by variant.
SAMPLERS_REGISTRY = NamedRegistry("sampler")
#: Evaluation metrics, optionally scoped to some variants.
METRICS_REGISTRY = VariantRegistry("metric")

MODELS = MODELS_REGISTRY.items
LOSSES = LOSSES_REGISTRY.items
SAMPLERS = SAMPLERS_REGISTRY.items
METRICS = METRICS_REGISTRY.items
