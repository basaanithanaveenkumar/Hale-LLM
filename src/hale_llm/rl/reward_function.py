"""Abstract reward: scores the completions of one rollout batch."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

import torch


class RewardFunction(ABC):
    """Maps ``(prompt, completion)`` pairs to scalar rewards.

    Completions arrive grouped: ``completions[g * group_size:(g + 1) * group_size]`` all answer
    ``prompts[g]``. Rewards that need the whole group (e.g. majority vote) rely on that layout.
    """

    #: whether the reward reads ground-truth answers (False = label-free / self-reward).
    uses_labels: bool = True

    @abstractmethod
    def __call__(
        self, prompts: Sequence[str], completions: Sequence[str], group_size: int
    ) -> torch.Tensor:
        """Return a float tensor of ``len(completions)`` rewards."""

    @staticmethod
    def _check_layout(prompts: Sequence[str], completions: Sequence[str], group_size: int) -> None:
        if len(completions) != len(prompts) * group_size:
            from hale_llm.rl.online_rl_error import OnlineRLError

            raise OnlineRLError(
                f"{len(completions)} completions for {len(prompts)} prompts "
                f"× group size {group_size}"
            )
