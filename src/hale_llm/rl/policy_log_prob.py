"""Abstract per-token log-probability of completions under a policy, for each model family."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

import torch


class PolicyLogProb(ABC):
    """Computes ``log π(y_k | x, …)`` for every completion token (shape ``(N, L_completion)``).

    Autoregressive models give it exactly. Masked-diffusion models have no tractable sequence
    likelihood, so their subclass returns an estimate. The trainer only sees this interface.

    Args:
        tokenizer: provides ``mask_token_id`` and ``encode``/``decode``.
    """

    def __init__(self, tokenizer) -> None:
        self.tokenizer = tokenizer

    @abstractmethod
    def __call__(
        self,
        model: torch.nn.Module,
        prompt_ids: torch.Tensor,
        completion_ids: torch.Tensor,
        stochastic: bool = True,
    ) -> torch.Tensor:
        """Per-token log-probs; ``stochastic=False`` disables random masking (evaluation)."""

    @torch.no_grad()
    def answer_distribution(
        self, model: torch.nn.Module, prompt: str, candidates: Sequence[str]
    ) -> torch.Tensor:
        """Normalised probability of each candidate completion (they must share one length).

        Args:
            model: the policy.
            prompt: one prompt.
            candidates: every completion to score, e.g. all two-digit answers.

        Returns:
            ``(len(candidates),)`` probabilities summing to 1 over the candidates.
        """
        prompt_ids = torch.tensor([self.tokenizer.encode(prompt)]).expand(len(candidates), -1)
        completion_ids = torch.tensor([self.tokenizer.encode(c) for c in candidates])
        model.eval()
        sequence_logp = self(model, prompt_ids, completion_ids, stochastic=False).sum(-1)
        return sequence_logp.softmax(0)
