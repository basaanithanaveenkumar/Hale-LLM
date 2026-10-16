"""Exact completion log-probabilities of an autoregressive (next-token) policy."""

from __future__ import annotations

import torch

from hale_llm.rl.policy_log_prob import PolicyLogProb
from hale_llm.rl.rl_registry import register_log_prob


@register_log_prob("autoregressive")
class AutoregressiveLogProb(PolicyLogProb):
    """``log π(y_k | x, y_<k)`` from one teacher-forced forward pass over ``[x ; y]``.

    The logits at position ``i`` predict token ``i + 1``, so completion token ``k`` (at sequence
    position ``L_p + k``) is read from the logits at ``L_p + k - 1``. The chain rule makes the sum
    over ``k`` the exact sequence log-likelihood.
    """

    def __call__(
        self,
        model: torch.nn.Module,
        prompt_ids: torch.Tensor,
        completion_ids: torch.Tensor,
        stochastic: bool = True,
    ) -> torch.Tensor:
        del stochastic  # exact: nothing random
        sequence = torch.cat([prompt_ids, completion_ids], dim=1)
        start = prompt_ids.shape[1]
        logits = model({"input_ids": sequence, "attention_mask": torch.ones_like(sequence)})
        log_probs = logits[:, start - 1 : -1].float().log_softmax(-1)
        return log_probs.gather(-1, completion_ids[..., None])[..., 0]
