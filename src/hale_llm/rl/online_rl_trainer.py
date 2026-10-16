"""The online RL loop: the policy samples its own data, is scored, and updates on it (GRPO)."""

from __future__ import annotations

import copy
from collections.abc import Sequence

import torch
from loguru import logger

from hale_llm.rl.group_relative_advantage import GroupRelativeAdvantage
from hale_llm.rl.grpo_objective import GRPOObjective
from hale_llm.rl.online_rl_config import GRPOSection, RolloutSection
from hale_llm.rl.policy_log_prob import PolicyLogProb
from hale_llm.rl.reward_function import RewardFunction
from hale_llm.rl.rl_step_stats import RLStepStats
from hale_llm.rl.rollout_batch import RolloutBatch


class OnlineRLTrainer:
    """GRPO for any registered variant that has a sampler and a :class:`PolicyLogProb`.

    One :meth:`step` is the whole online loop:

    1. **collect**: sample ``G`` completions per prompt from the *current* policy (on-policy data)
       and freeze their log-probs as ``π_old``;
    2. **score**: reward every completion (verifier or self-reward);
    3. **normalise**: group-relative advantages (no critic);
    4. **update**: ``inner_epochs`` gradient steps on the clipped surrogate + KL to ``π_ref``.

    Args:
        policy: the model being trained (updated in place).
        tokenizer: char tokenizer shared with the sampler.
        variant: registry name of the model family (selects the sampler).
        reward: scores completions.
        log_prob: per-token log-probabilities for this variant.
        rollout: sampling settings.
        grpo: optimisation settings.
        completion_length: tokens per completion (fixed-length answers).
        reference: frozen KL anchor; defaults to a copy of ``policy`` at construction.
    """

    def __init__(
        self,
        policy: torch.nn.Module,
        tokenizer,
        variant: str,
        reward: RewardFunction,
        log_prob: PolicyLogProb,
        rollout: RolloutSection,
        grpo: GRPOSection,
        completion_length: int,
        reference: torch.nn.Module | None = None,
    ) -> None:
        from hale_core.registry import get_sampler

        self.policy = policy
        self.tokenizer = tokenizer
        self.variant = variant
        self.reward = reward
        self.log_prob = log_prob
        self.rollout = rollout
        self.grpo = grpo
        self.completion_length = completion_length
        self.sampler = get_sampler(variant)
        self.reference = (reference or copy.deepcopy(policy)).eval().requires_grad_(False)
        self.advantage = GroupRelativeAdvantage(normalize_std=grpo.normalize_std)
        self.objective = GRPOObjective(clip_eps=grpo.clip_eps, kl_coef=grpo.kl_coef)
        self.optimizer = torch.optim.AdamW(policy.parameters(), lr=grpo.lr, weight_decay=0.0)
        self.steps_done = 0

    # ------------------------------------------------------------------ 1-3: collect and score
    @torch.no_grad()
    def collect(self, prompts: Sequence[str]) -> RolloutBatch:
        """Sample, decode, reward and normalise one group per prompt."""
        G = self.rollout.group_size
        prompt_ids = torch.tensor([self.tokenizer.encode(p) for p in prompts]).repeat_interleave(
            G, 0
        )
        self.policy.eval()
        kwargs = (
            {"sampling_steps": self.rollout.sampling_steps}
            if self.variant != "autoregressive"
            else {}
        )
        sequences = self.sampler(
            self.policy,
            self.tokenizer,
            prompt_ids,
            self.completion_length,
            temperature=self.rollout.temperature,
            **kwargs,
        )
        completion_ids = sequences[:, prompt_ids.shape[1] :]
        completions = [self.tokenizer.decode(c) for c in completion_ids]
        rewards = self.reward(list(prompts), completions, G)
        return RolloutBatch(
            list(prompts),
            prompt_ids,
            completion_ids,
            completions,
            rewards,
            self.advantage(rewards, G),
            G,
        )

    # ------------------------------------------------------------------ 4: update
    def update(self, batch: RolloutBatch) -> RLStepStats:
        """Run ``inner_epochs`` clipped-surrogate updates on one rollout batch."""
        with torch.no_grad():
            self.policy.eval()
            old = self.log_prob(
                self.policy, batch.prompt_ids, batch.completion_ids, stochastic=False
            )
            ref = self.log_prob(
                self.reference, batch.prompt_ids, batch.completion_ids, stochastic=False
            )
        self.policy.train()
        for _ in range(self.grpo.inner_epochs):
            new = self.log_prob(self.policy, batch.prompt_ids, batch.completion_ids)
            terms = self.objective(new, old, batch.advantages, ref_logp=ref)
            self.optimizer.zero_grad(set_to_none=True)
            terms.loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.grpo.max_grad_norm)
            self.optimizer.step()
        self.policy.eval()
        self.steps_done += 1
        G = batch.group_size
        unique = sum(
            len(set(batch.completions[i : i + G])) for i in range(0, len(batch.completions), G)
        )
        return RLStepStats(
            step=self.steps_done,
            reward_mean=float(batch.rewards.mean()),
            zero_variance_fraction=GroupRelativeAdvantage.zero_variance_fraction(batch.rewards, G),
            unique_answers=unique / len(batch.prompts),
            policy_loss=terms.policy_loss,
            kl=terms.kl,
            clip_fraction=terms.clip_fraction,
        )

    def step(self, prompts: Sequence[str]) -> tuple[RLStepStats, RolloutBatch]:
        """One full online iteration on ``prompts``; returns the stats and the rollouts used."""
        batch = self.collect(prompts)
        stats = self.update(batch)
        if stats.step % 100 == 0:
            logger.info(
                "rl step {}: reward {:.3f}, zero-var {:.2f}, kl {:.4f}",
                stats.step,
                stats.reward_mean,
                stats.zero_variance_fraction,
                stats.kl,
            )
        return stats, batch
