"""The training loss became NaN or infinite."""

from __future__ import annotations

from hale_llm.exceptions.training_error import TrainingError


class NonFiniteLossError(TrainingError, FloatingPointError):
    """The training loss became NaN or infinite.

    Carries the step so the run can be resumed from the previous checkpoint with a lower learning
    rate. Also a :class:`FloatingPointError` (the original type).

    Args:
        step: global optimisation step at which the loss was non-finite.
    """

    def __init__(self, step: int) -> None:
        self.step = step
        super().__init__(f"non-finite loss at step {step}")
