"""The clipped GRPO surrogate with a KL penalty to a frozen reference policy."""

from __future__ import annotations

import torch

from hale_llm.rl.grpo_loss_terms import GRPOLossTerms


class GRPOObjective:
    r"""Per-token PPO-clip surrogate on group-relative advantages, plus ``β · KL(π ‖ π_ref)``.

    .. math::
        \mathcal{L} = -\frac{1}{N}\sum_i \frac{1}{|y_i|}\sum_k \Big[\min\big(\rho_{ik} A_i,\;
        \mathrm{clip}(\rho_{ik}, 1-\epsilon, 1+\epsilon) A_i\big) - \beta\, \mathbb{D}_{ik}\Big],
        \quad \rho_{ik} = \frac{\pi_\theta(y_{ik}\mid\cdot)}{\pi_{\text{old}}(y_{ik}\mid\cdot)}

    The KL uses the non-negative, unbiased "k3" estimator
    :math:`\mathbb{D} = e^{\log\pi_{ref}-\log\pi_\theta} - (\log\pi_{ref}-\log\pi_\theta) - 1`.

    Args:
        clip_eps: PPO clip range ε (the ratio is held inside ``[1-ε, 1+ε]``).
        kl_coef: β; 0 disables the reference model.
    """

    def __init__(self, clip_eps: float = 0.2, kl_coef: float = 0.02) -> None:
        self.clip_eps = clip_eps
        self.kl_coef = kl_coef

    def __call__(
        self,
        new_logp: torch.Tensor,
        old_logp: torch.Tensor,
        advantages: torch.Tensor,
        ref_logp: torch.Tensor | None = None,
        completion_mask: torch.Tensor | None = None,
    ) -> GRPOLossTerms:
        """Loss over ``(N, L)`` per-token log-probs; ``advantages`` is ``(N,)``.

        Args:
            new_logp: log-probs under the policy being optimised (with gradient).
            old_logp: log-probs under the policy that sampled the rollouts (no gradient).
            advantages: one advantage per completion, broadcast over its tokens.
            ref_logp: log-probs under the frozen reference policy (for the KL term).
            completion_mask: 1 for real completion tokens (default: all).
        """
        mask = torch.ones_like(new_logp) if completion_mask is None else completion_mask.float()
        ratio = (new_logp - old_logp).exp()
        adv = advantages[:, None]
        surrogate = torch.min(ratio * adv, ratio.clamp(1 - self.clip_eps, 1 + self.clip_eps) * adv)
        per_token = -surrogate
        kl = torch.zeros_like(new_logp)
        if ref_logp is not None and self.kl_coef > 0:
            log_ratio = ref_logp - new_logp
            kl = log_ratio.exp() - log_ratio - 1
            per_token = per_token + self.kl_coef * kl
        per_sequence = (per_token * mask).sum(1) / mask.sum(1).clamp(min=1)
        clipped = ((ratio - 1).abs() > self.clip_eps).float()
        tokens = mask.sum().clamp(min=1)
        return GRPOLossTerms(
            loss=per_sequence.mean(),
            policy_loss=float((-(surrogate * mask)).sum().detach() / tokens),
            kl=float((kl * mask).sum().detach() / tokens),
            clip_fraction=float((clipped * mask).sum().detach() / tokens),
            ratio_mean=float((ratio * mask).sum().detach() / tokens),
        )
