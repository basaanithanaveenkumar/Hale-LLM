"""One-step mean-field log-probability estimate for masked-diffusion policies (d1 / diffu-GRPO)."""

from __future__ import annotations

import torch

from hale_llm.rl.policy_log_prob import PolicyLogProb
from hale_llm.rl.rl_registry import register_log_prob


@register_log_prob("diffusion")
class MaskedDiffusionLogProb(PolicyLogProb):
    """Estimate ``log π(y_k | x)`` with a single forward pass at full masking.

    A masked-diffusion LM generates by unmasking in an order chosen at sampling time, so
    ``log π(y | x)`` sums over all orders and is intractable. d1 (Zhao et al., 2025) uses a cheap
    **mean-field** surrogate. Mask *every* completion token, run the model once at ``t = 1``, and
    read each token's log-probability independently, so the sequence log-prob is approximately the
    sum over tokens. During training, a random fraction ``prompt_mask_prob`` of the *prompt*
    tokens is also masked. This gives each inner GRPO epoch a different view of the same rollout,
    which acts as a regulariser and lets several gradient steps reuse one batch of samples.

    Args:
        tokenizer: provides ``mask_token_id``.
        prompt_mask_prob: fraction of prompt tokens masked when ``stochastic=True``.
    """

    def __init__(self, tokenizer, prompt_mask_prob: float = 0.15) -> None:
        super().__init__(tokenizer)
        self.prompt_mask_prob = prompt_mask_prob

    def __call__(
        self,
        model: torch.nn.Module,
        prompt_ids: torch.Tensor,
        completion_ids: torch.Tensor,
        stochastic: bool = True,
    ) -> torch.Tensor:
        mask_id = self.tokenizer.mask_token_id
        start = prompt_ids.shape[1]
        x = torch.cat([prompt_ids, torch.full_like(completion_ids, mask_id)], dim=1)
        if stochastic and self.prompt_mask_prob > 0:
            hide = torch.rand(prompt_ids.shape, device=x.device) < self.prompt_mask_prob
            x[:, :start] = torch.where(hide, torch.full_like(prompt_ids, mask_id), prompt_ids)
        t = torch.ones(x.shape[0], device=x.device)  # fully masked completion
        logits = model({"x_t": x, "t": t, "attention_mask": None})[:, start:].float()
        logits[..., mask_id] = -float("inf")  # the model never emits <mask>
        return logits.log_softmax(-1).gather(-1, completion_ids[..., None])[..., 0]
