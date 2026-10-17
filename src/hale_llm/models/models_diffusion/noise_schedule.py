"""Strategy interface for the forward-masking schedule of masked-diffusion LMs."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch


class NoiseSchedule(ABC):
    """How much of a sequence is masked at diffusion time ``t`` ∈ (0, 1] (Strategy pattern).

    The loss and the sampler of :mod:`hale_llm.models.models_diffusion` only talk to this
    interface, so a cosine or polynomial schedule is a new subclass, not an edit of the loss or
    sampler. The continuous-time MDLM objective weights each masked token by
    ``-alpha'(t) / (1 - alpha(t))``; :meth:`loss_weight` returns that factor directly.
    """

    @abstractmethod
    def alpha(self, t: torch.Tensor) -> torch.Tensor:
        """Probability that a token is still *unmasked* at time ``t``."""

    @abstractmethod
    def mask_probability(self, t: torch.Tensor) -> torch.Tensor:
        """Probability that a token is masked at time ``t`` (``1 - alpha(t)``)."""

    @abstractmethod
    def loss_weight(self, t: torch.Tensor, eps: float) -> torch.Tensor:
        """ELBO weight ``-alpha'(t) / (1 - alpha(t))`` of a masked token, with ``t`` floored at ``eps``."""

    @abstractmethod
    def reveal_probability(self, t_cur: torch.Tensor, t_next: torch.Tensor) -> torch.Tensor:
        """Probability that a token masked at ``t_cur`` is revealed by ``t_next < t_cur``."""
