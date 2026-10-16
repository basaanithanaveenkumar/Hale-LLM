"""One batch of self-generated experience: G completions per prompt, scored and normalised."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class RolloutBatch:
    """Rollouts in grouped layout: rows ``g*G … g*G+G-1`` answer ``prompts[g]``.

    Attributes:
        prompts: the ``P`` prompt strings.
        prompt_ids: ``(P*G, L_p)`` prompt tokens repeated per sample.
        completion_ids: ``(P*G, L_c)`` sampled completion tokens.
        completions: decoded completions, ``P*G`` strings.
        rewards: ``(P*G,)`` rewards.
        advantages: ``(P*G,)`` group-relative advantages.
        group_size: ``G``.
    """

    prompts: list[str]
    prompt_ids: torch.Tensor
    completion_ids: torch.Tensor
    completions: list[str]
    rewards: torch.Tensor
    advantages: torch.Tensor
    group_size: int

    def group(self, index: int) -> tuple[str, list[str], list[float], list[float]]:
        """``(prompt, completions, rewards, advantages)`` of group ``index``."""
        sl = slice(index * self.group_size, (index + 1) * self.group_size)
        return (
            self.prompts[index],
            self.completions[sl],
            self.rewards[sl].tolist(),
            self.advantages[sl].tolist(),
        )
