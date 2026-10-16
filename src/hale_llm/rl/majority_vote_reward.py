"""Label-free self-reward: agree with the group's majority answer (Test-Time RL, TTRL)."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence

import torch

from hale_llm.rl.reward_function import RewardFunction
from hale_llm.rl.rl_registry import register_reward


@register_reward("majority_vote")
class MajorityVoteReward(RewardFunction):
    """Reward 1 for completions equal to the most common answer in their group, else 0.

    No labels are read: the policy's own consensus is the pseudo-label. This sharpens whatever
    the model already believes. It helps when the majority is usually right, and it locks in
    errors when it is not (shown in the RL notebook). Ties go to the answer seen first.

    Args:
        answers: accepted for a uniform constructor signature; ignored.
    """

    uses_labels = False

    def __init__(self, answers: Mapping[str, str] | None = None) -> None:
        del answers

    def __call__(
        self, prompts: Sequence[str], completions: Sequence[str], group_size: int
    ) -> torch.Tensor:
        self._check_layout(prompts, completions, group_size)
        rewards = []
        for g in range(len(prompts)):
            group = [c.strip() for c in completions[g * group_size : (g + 1) * group_size]]
            majority = Counter(group).most_common(1)[0][0]
            rewards += [float(c == majority) for c in group]
        return torch.tensor(rewards)

    @staticmethod
    def pseudo_labels(completions: Sequence[str], group_size: int) -> list[str]:
        """The majority answer of every group (for plots and animations)."""
        groups = [completions[i : i + group_size] for i in range(0, len(completions), group_size)]
        return [Counter(c.strip() for c in g).most_common(1)[0][0] for g in groups]
