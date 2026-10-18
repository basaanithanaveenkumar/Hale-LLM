"""Strategy interface and registry for the ``kappa`` schedules of the masking path."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import TypeVar

import torch
from hale_core.registry.base import NamedRegistry

T = TypeVar("T")

#: ``model.fm_schedule`` -> :class:`KappaSchedule` subclass.
KAPPA_SCHEDULES = NamedRegistry("kappa schedule")


def register_kappa_schedule(name: str) -> Callable[[type[T]], type[T]]:
    """Class decorator registering a :class:`KappaSchedule` under ``name``."""

    def decorator(cls: type[T]) -> type[T]:
        KAPPA_SCHEDULES.add(name, cls)
        return cls

    return decorator


class KappaSchedule(ABC):
    """Strategy: probability ``kappa(t)`` that a token is already data at time ``t``.

    Contract: ``kappa(0) = 0``, ``kappa(1) = 1``, monotonically increasing.
    """

    @abstractmethod
    def kappa(self, t: torch.Tensor | float) -> torch.Tensor | float:
        """Return ``kappa(t)``; a tensor in gives a tensor out, a float gives a float."""

    @abstractmethod
    def rate(self, t: torch.Tensor) -> torch.Tensor:
        """Return ``kappa'(t) / (1 - kappa(t))`` for ``t`` strictly below 1."""
