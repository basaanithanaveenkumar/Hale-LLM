"""How the weight of the multi-token-prediction loss changes during training."""

from __future__ import annotations


class MTPLossSchedule:
    """A step schedule for lambda, the weight of the MTP loss (DeepSeek-V3: 0.3, then 0.1).

    ``first`` applies until ``switch_fraction`` of the training has passed, ``final`` after that. With
    ``final=None`` the weight is constant. The input is progress through the run, ``step / total_steps``
    (steps stand in for tokens: every step sees about the same number of tokens), so the schedule is the same
    for every depth and a resumed run lands on the same value as one that never stopped.

    Example with ``first=0.3, final=0.1, switch_fraction=0.68``::

        weight_at(0.0)  -> 0.3        weight_at(0.679) -> 0.3
        weight_at(0.68) -> 0.1        weight_at(0.99)  -> 0.1
    """

    def __init__(self, first: float, final: float | None, switch_fraction: float) -> None:
        self.first = first
        self.final = final
        self.switch_fraction = switch_fraction

    @classmethod
    def from_config(cls, model_cfg) -> MTPLossSchedule:
        """Build the schedule from ``model.mtp_loss_weight``, ``mtp_loss_weight_final``, ``mtp_switch_fraction``."""
        return cls(
            model_cfg.mtp_loss_weight,
            model_cfg.mtp_loss_weight_final,
            model_cfg.mtp_switch_fraction,
        )

    def weight_at(self, progress: float) -> float:
        """Lambda for a step that is ``progress`` (0 to 1) of the way through training."""
        if self.final is None or progress < self.switch_fraction:
            return self.first
        return self.final
