"""One name may map to several implementations, picked by model variant at run time."""

from __future__ import annotations

from typing import Any

from loguru import logger

VariantKey = frozenset[str] | None


class VariantRegistry:
    """Register several implementations under one name, each scoped to some variants.

    ``resolve(name, variant)`` picks, in order:

    1. an entry whose variant set contains ``variant``,
    2. an entry registered with ``variants=None`` (the generic fallback),
    3. ``None`` when neither exists (the metric does not apply to this variant).
    """

    def __init__(self, kind: str) -> None:
        self.kind = kind
        self.items: dict[str, list[tuple[VariantKey, type]]] = {}

    def add(
        self,
        name: str,
        cls: type,
        *,
        variants: tuple[str, ...] | None = None,
        attr: str | None = "metric_name",
    ) -> type:
        """Register ``cls`` as ``name`` for ``variants`` (``None`` = every variant)."""
        key: VariantKey = frozenset(variants) if variants is not None else None
        entries = self.items.setdefault(name, [])
        if any(existing == key for existing, _ in entries):
            raise ValueError(f"{self.kind} {name!r} already registered for variants={variants}")
        entries.append((key, cls))
        if attr is not None:
            setattr(cls, attr, name)
        logger.debug("registered {} {} variants={}", self.kind, name, variants)
        return cls

    def resolve(self, name: str, variant: str) -> type | None:
        """Return the class for ``name`` and ``variant`` (see the class docstring)."""
        entries = self.items.get(name)
        if entries is None:
            logger.error("unknown {} {!r}; registered={}", self.kind, name, sorted(self.items))
            raise KeyError(f"unknown {self.kind} {name!r}; registered={sorted(self.items)}")
        specific = [cls for keys, cls in entries if keys is not None and variant in keys]
        generic = [cls for keys, cls in entries if keys is None]
        return (specific or generic or [None])[0]

    def instantiate(self, names: list[str], variant: str) -> list[Any]:
        """Build one object per name, skipping names that do not apply to ``variant``."""
        built: list[Any] = []
        for name in names:
            cls = self.resolve(name, variant)
            if cls is None:
                logger.info("skipping {} {} (not defined for variant {})", self.kind, name, variant)
                continue
            built.append(cls())
        return built
