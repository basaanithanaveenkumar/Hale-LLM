"""GRPO's critic-free advantage: each reward compared with the other samples of the same prompt."""

from __future__ import annotations

import torch

from hale_llm.rl.online_rl_error import OnlineRLError


class GroupRelativeAdvantage:
    """``A_i = (r_i − mean(r_group)) / (std(r_group) + eps)``.

    PPO learns a value network to estimate how good a prompt is. GRPO instead samples ``G``
    completions per prompt and uses their mean reward as the baseline. That needs no critic, and
    the baseline is exactly on-policy. A group whose rewards are all equal (all right or all
    wrong) gets **zero** advantage and so contributes no gradient. The fraction of such groups is
    the key exploration diagnostic.

    Args:
        normalize_std: divide by the group std (GRPO); ``False`` gives the Dr. GRPO variant, which
            avoids up-weighting near-deterministic groups.
        eps: numerical floor for the std.
    """

    def __init__(self, normalize_std: bool = True, eps: float = 1e-4) -> None:
        self.normalize_std = normalize_std
        self.eps = eps

    def __call__(self, rewards: torch.Tensor, group_size: int) -> torch.Tensor:
        """Advantages with the same shape as ``rewards`` (grouped layout).

        Raises:
            OnlineRLError: if the reward count is not a multiple of ``group_size``.
        """
        if rewards.numel() % group_size:
            raise OnlineRLError(
                f"{rewards.numel()} rewards do not split into groups of {group_size}"
            )
        grouped = rewards.view(-1, group_size)
        centred = grouped - grouped.mean(dim=1, keepdim=True)
        if self.normalize_std:
            centred = centred / (grouped.std(dim=1, keepdim=True) + self.eps)
        return centred.reshape(-1)

    @staticmethod
    def zero_variance_fraction(rewards: torch.Tensor, group_size: int) -> float:
        """Fraction of groups with identical rewards (no learning signal)."""
        grouped = rewards.view(-1, group_size)
        return float((grouped.max(dim=1).values == grouped.min(dim=1).values).float().mean())
