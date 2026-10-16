"""Verifiable reward: 1 if the completion equals the known answer, else 0 (RLVR)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch

from hale_llm.rl.reward_function import RewardFunction
from hale_llm.rl.rl_registry import register_reward


@register_reward("exact_match")
class ExactMatchReward(RewardFunction):
    """Rule-based reward with a checker, as in RL from verifiable rewards (DeepSeek-R1, Tülu 3).

    Args:
        answers: ``prompt -> reference answer``.
    """

    uses_labels = True

    def __init__(self, answers: Mapping[str, str]) -> None:
        self.answers = dict(answers)

    def __call__(
        self, prompts: Sequence[str], completions: Sequence[str], group_size: int
    ) -> torch.Tensor:
        self._check_layout(prompts, completions, group_size)
        gold = [self.answers[p] for p in prompts for _ in range(group_size)]
        return torch.tensor([float(c.strip() == g) for c, g in zip(completions, gold, strict=True)])
