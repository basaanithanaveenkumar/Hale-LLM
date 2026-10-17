"""The log-linear schedule ``alpha(t) = 1 - t`` used by MDLM and this repo's diffusion variant."""

from __future__ import annotations

import torch

from hale_llm.models.models_diffusion.noise_schedule import NoiseSchedule


class LogLinearSchedule(NoiseSchedule):
    """``alpha(t) = 1 - t``: a token is masked with probability ``t``.

    Every method returns the exact expression the loss and sampler used before the schedule was
    extracted (``t`` itself rather than ``1 - (1 - t)``, and so on), so results are bit-identical.
    """

    def alpha(self, t: torch.Tensor) -> torch.Tensor:
        """``1 - t``."""
        return 1.0 - t

    def mask_probability(self, t: torch.Tensor) -> torch.Tensor:
        """``t``."""
        return t

    def loss_weight(self, t: torch.Tensor, eps: float) -> torch.Tensor:
        # -alpha'(t) / (1 - alpha(t)) = 1 / t
        """``1 / max(t, eps)``."""
        return 1.0 / t.clamp(min=eps)

    def reveal_probability(self, t_cur: torch.Tensor, t_next: torch.Tensor) -> torch.Tensor:
        # P(revealed by t_next | masked at t_cur) = (alpha(t_next) - alpha(t_cur)) / (1 - alpha(t_cur))
        """``(t_cur - t_next) / t_cur``, clamped to [0, 1]."""
        return ((t_cur - t_next) / t_cur.clamp(min=1e-8)).clamp(0, 1)
