"""Name -> training-backend lookup (``train.trainer`` in YAML)."""

from __future__ import annotations

from collections.abc import Callable

from hale_llm.registry import NamedRegistry

from hale_llm.training.backends.trainer_backend import TrainerBackend

TRAINER_BACKENDS = NamedRegistry("trainer_backend")


def register_trainer_backend(name: str) -> Callable[[type[TrainerBackend]], type[TrainerBackend]]:
    """Class decorator: make ``name`` selectable as ``train.trainer``."""

    def decorator(cls: type[TrainerBackend]) -> type[TrainerBackend]:
        cls.name = name
        TRAINER_BACKENDS.add(name, cls)
        return cls

    return decorator


def get_trainer_backend(name: str) -> TrainerBackend:
    """Instantiate the backend registered under ``name`` (KeyError lists the options)."""
    return TRAINER_BACKENDS.get(name)()
