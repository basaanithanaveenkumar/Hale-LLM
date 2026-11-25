"""Decorator-based plugin registries (models, losses, samplers, metrics)."""

from hale_llm.registry.decorators import (
    get_loss,
    get_sampler,
    get_variant,
    instantiate_metrics,
    register_loss,
    register_metric,
    register_sampler,
    register_variant,
)
from hale_llm.registry.named_registry import NamedRegistry
from hale_llm.registry.registries import (
    LOSSES,
    METRICS,
    MODELS,
    SAMPLERS,
)
from hale_llm.registry.variant_registry import VariantRegistry

__all__ = [
    "LOSSES",
    "METRICS",
    "MODELS",
    "SAMPLERS",
    "NamedRegistry",
    "VariantRegistry",
    "get_loss",
    "get_sampler",
    "get_variant",
    "instantiate_metrics",
    "register_loss",
    "register_metric",
    "register_sampler",
    "register_variant",
]
