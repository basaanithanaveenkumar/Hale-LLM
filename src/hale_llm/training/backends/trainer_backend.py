"""Interface every training backend implements."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from hale_llm.config.schema import RunConfig


class TrainerBackend(ABC):
    """One way of running a training job (lean loop, full trainer, TorchTitan, ...).

    The CLI only talks to this interface, so a new backend is one subclass plus a
    ``@register_trainer_backend`` decorator; no existing code changes.
    """

    #: Name used in ``train.trainer``.
    name: str = "base"

    def validate(self, cfg: RunConfig) -> None:  # noqa: B027  (optional hook)
        """Raise if ``cfg`` asks for something this backend cannot do. Default: accept."""

    @abstractmethod
    def run(self, cfg: RunConfig, tokenizer=None) -> Any:
        """Train according to ``cfg`` and return the backend's result."""
