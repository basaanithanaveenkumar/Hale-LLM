"""Interface for ways of stretching RoPE to longer contexts than it was trained on."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch


class RopeScaling(ABC):
    """Computes the rotation frequencies of RoPE, possibly adjusted for a longer context.

    RoPE rotates channel pair ``m`` by ``position * freq[m]`` radians. A model trained
    up to length ``L`` has only seen angles reachable within ``L`` positions; running it
    at ``s * L`` positions shows it unseen angles. Scaling strategies change the
    frequencies so those angles look familiar again.
    """

    #: Name used in ``model.rope_scaling``.
    name: str = "base"

    @abstractmethod
    def compute(
        self,
        head_dim: int,
        theta: float,
        factor: float,
        original_length: int,
    ) -> tuple[torch.Tensor, float]:
        """Return ``(inverse_frequencies, attention_scale)``.

        ``inverse_frequencies`` has ``head_dim // 2`` entries. ``attention_scale`` multiplies
        the rotated queries and keys (1.0 means unchanged).
        """

    @staticmethod
    def base_frequencies(head_dim: int, theta: float) -> torch.Tensor:
        """Plain RoPE: ``theta ** (-m / half)`` for channel pair ``m``."""
        half = head_dim // 2
        return 1.0 / (theta ** (torch.arange(half, dtype=torch.float32) / half))
