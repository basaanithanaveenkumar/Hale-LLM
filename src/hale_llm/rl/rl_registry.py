"""Registries of the online RL package: reward functions and policy log-probability estimators.

Both are chosen by name from YAML (``reward.type``, the model ``variant``), so a new reward or a new
model family plugs in by registering a class; the trainer never changes (open/closed principle).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

from hale_core.registry.base import NamedRegistry

T = TypeVar("T")

#: ``reward.type`` -> :class:`RewardFunction` subclass.
REWARDS = NamedRegistry("reward")
#: model ``variant`` -> :class:`PolicyLogProb` subclass.
LOG_PROB_ESTIMATORS = NamedRegistry("policy log-prob estimator")


def register_reward(name: str) -> Callable[[type[T]], type[T]]:
    """Class decorator registering a reward function under ``name``."""

    def decorator(cls: type[T]) -> type[T]:
        REWARDS.add(name, cls)
        return cls

    return decorator


def register_log_prob(variant: str) -> Callable[[type[T]], type[T]]:
    """Class decorator registering the log-probability estimator of model ``variant``."""

    def decorator(cls: type[T]) -> type[T]:
        LOG_PROB_ESTIMATORS.add(variant, cls)
        return cls

    return decorator
