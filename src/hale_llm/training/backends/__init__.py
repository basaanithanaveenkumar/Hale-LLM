"""Training backends. Importing this package registers ``lean``, ``full`` and ``torchtitan``."""

from hale_llm.training.backends.full_backend import FullTrainerBackend
from hale_llm.training.backends.lean_backend import LeanTrainerBackend
from hale_llm.training.backends.registry import (
    TRAINER_BACKENDS,
    get_trainer_backend,
    register_trainer_backend,
)
from hale_llm.training.backends.titan_backend import TorchTitanBackend
from hale_llm.training.backends.titan_command_builder import TorchTitanCommandBuilder
from hale_llm.training.backends.titan_errors import TorchTitanUnavailableError
from hale_llm.training.backends.trainer_backend import TrainerBackend

__all__ = [
    "TRAINER_BACKENDS",
    "FullTrainerBackend",
    "LeanTrainerBackend",
    "TorchTitanBackend",
    "TorchTitanCommandBuilder",
    "TorchTitanUnavailableError",
    "TrainerBackend",
    "get_trainer_backend",
    "register_trainer_backend",
]
