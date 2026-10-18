"""Linear ``kappa`` schedule."""

from __future__ import annotations

import torch

from hale_llm.models.models_block_flow_matching.kappa_schedule import (
    KappaSchedule,
    register_kappa_schedule,
)


@register_kappa_schedule("linear")
class LinearKappa(KappaSchedule):
    """Strategy ``kappa(t) = t``: tokens are revealed at a constant pace."""

    def kappa(self, t: torch.Tensor | float) -> torch.Tensor | float:
        """Return ``t`` unchanged."""
        return t

    def rate(self, t: torch.Tensor) -> torch.Tensor:
        """Return ``1 / (1 - t)``."""
        return 1.0 / (1.0 - t)
