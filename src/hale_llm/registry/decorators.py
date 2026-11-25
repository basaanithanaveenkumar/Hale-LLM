"""Decorators that add a class or function to a registry, and the matching lookups."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar

from hale_llm.registry.registries import (
    LOSSES_REGISTRY,
    METRICS_REGISTRY,
    MODELS_REGISTRY,
    SAMPLERS_REGISTRY,
)

T = TypeVar("T")


def register_variant(name: str) -> Callable[[type[T]], type[T]]:
    """Class decorator: make a model selectable as ``variant: <name>``."""

    def decorator(cls: type[T]) -> type[T]:
        MODELS_REGISTRY.add(name, cls)
        cls.variant_name = name  # type: ignore[attr-defined]
        return cls

    return decorator


def register_loss(name: str) -> Callable[[T], T]:
    """Function decorator: register ``loss(model, batch) -> scalar`` for a variant."""

    def decorator(fn: T) -> T:
        LOSSES_REGISTRY.add(name, fn)
        return fn

    return decorator


def register_sampler(name: str) -> Callable[[T], T]:
    """Function decorator: register a text sampler for a variant."""

    def decorator(fn: T) -> T:
        SAMPLERS_REGISTRY.add(name, fn)
        return fn

    return decorator


def register_metric(
    name: str, *, variants: tuple[str, ...] | None = None
) -> Callable[[type[T]], type[T]]:
    """Class decorator: register a metric, optionally only for some variants."""

    def decorator(cls: type[T]) -> type[T]:
        return METRICS_REGISTRY.add(name, cls, variants=variants)

    return decorator


def get_variant(name: str) -> type:
    return MODELS_REGISTRY.get(name)


def get_loss(name: str) -> Callable[..., Any]:
    return LOSSES_REGISTRY.get(name)


def get_sampler(name: str) -> Callable[..., Any]:
    return SAMPLERS_REGISTRY.get(name)


def instantiate_metrics(names: list[str], variant: str) -> list[Any]:
    """Build the named metrics that apply to ``variant``."""
    return METRICS_REGISTRY.instantiate(names, variant)
