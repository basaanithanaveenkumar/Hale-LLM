"""One name maps to exactly one registered object."""

from __future__ import annotations

from typing import Any

from loguru import logger


class NamedRegistry:
    """Open/closed lookup table: add entries with :meth:`add`, never by editing callers."""

    def __init__(self, kind: str) -> None:
        self.kind = kind
        self.items: dict[str, Any] = {}

    def add(self, name: str, obj: Any) -> Any:
        """Register ``obj`` under ``name``; a second registration of a name is an error."""
        if name in self.items:
            logger.error("duplicate {} registration {}", self.kind, name)
            raise ValueError(f"{self.kind} {name!r} already registered: {self.items[name]}")
        self.items[name] = obj
        logger.debug("registered {} {}", self.kind, name)
        return obj

    def get(self, name: str) -> Any:
        """Return the object registered under ``name`` (KeyError lists the valid names)."""
        try:
            return self.items[name]
        except KeyError as error:
            logger.error("unknown {} {!r}; registered={}", self.kind, name, sorted(self.items))
            raise KeyError(
                f"unknown {self.kind} {name!r}; registered={sorted(self.items)}"
            ) from error

    def __contains__(self, name: str) -> bool:
        return name in self.items

    def __iter__(self):
        return iter(self.items)
