"""Facade: config → supervised warm start → online GRPO → recorded history."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import torch
from loguru import logger

from hale_llm.rl.online_rl_config import OnlineRLConfig
from hale_llm.rl.online_rl_error import OnlineRLError
from hale_llm.rl.online_rl_trainer import OnlineRLTrainer
from hale_llm.rl.policy_evaluator import PolicyEvaluator
from hale_llm.rl.rl_history import RLHistory
from hale_llm.rl.rl_registry import LOG_PROB_ESTIMATORS, REWARDS
from hale_llm.rl.tasks.arithmetic_task import ArithmeticTask


class OnlineRLExperiment:
    """Builds every part of an online RL run from one :class:`OnlineRLConfig`.

    Example:
        >>> path = "configs/rl/arithmetic_autoregressive.yaml"
        >>> exp = OnlineRLExperiment.from_yaml(path)  # doctest: +SKIP
        >>> history = exp.run(exp.warm_start())  # doctest: +SKIP

    Args:
        cfg: validated configuration.
    """

    def __init__(self, cfg: OnlineRLConfig) -> None:
        import hale_llm.models  # noqa: F401 - registers variants, samplers and losses
        import hale_llm.rl.autoregressive_log_prob  # noqa: F401 - registers estimators and rewards
        import hale_llm.rl.exact_match_reward  # noqa: F401
        import hale_llm.rl.majority_vote_reward  # noqa: F401
        import hale_llm.rl.masked_diffusion_log_prob  # noqa: F401
        from hale_llm.visualization.explainers.tokenizer import CharTokenizer

        self.cfg = cfg
        self.task = ArithmeticTask(cfg.task.max_operand, cfg.task.sft_fraction, seed=cfg.seed)
        self.tokenizer = CharTokenizer()
        self.evaluator = PolicyEvaluator(
            self.tokenizer, cfg.variant, self.task.completion_length, cfg.rollout.sampling_steps
        )

    @classmethod
    def from_yaml(
        cls, path: str | Path, overrides: dict[str, Any] | None = None
    ) -> OnlineRLExperiment:
        cfg = OnlineRLConfig.from_yaml(path)
        return cls(cfg.with_overrides(overrides) if overrides else cfg)

    # ------------------------------------------------------------------ parts
    def warm_start(self) -> torch.nn.Module:
        """Supervised fine-tuning on the ``sft_fraction`` of facts (the RL starting policy)."""
        from hale_llm.visualization.explainers.toy import train_toy

        run = train_toy(
            self.cfg.variant,
            self.task.sft_texts,
            model=self.cfg.model,
            steps=self.cfg.sft.steps,
            lr=self.cfg.sft.lr,
            batch_size=len(self.task.sft_texts),
            seed=self.cfg.seed,
        )
        logger.info(
            "warm start: {} {} steps, final loss {:.3f}",
            self.cfg.variant,
            self.cfg.sft.steps,
            run.losses[-1] if run.losses else float("nan"),
        )
        return run.model

    def log_prob(self):
        """The registered :class:`PolicyLogProb` for the configured variant."""
        if self.cfg.variant not in LOG_PROB_ESTIMATORS:
            raise OnlineRLError(f"no log-prob estimator registered for {self.cfg.variant!r}")
        cls = LOG_PROB_ESTIMATORS.get(self.cfg.variant)
        if self.cfg.variant == "diffusion":
            return cls(self.tokenizer, prompt_mask_prob=self.cfg.grpo.prompt_mask_prob)
        return cls(self.tokenizer)

    def reward(self):
        """The registered reward named by ``reward.type``."""
        if self.cfg.reward.type not in REWARDS:
            raise OnlineRLError(
                f"unknown reward {self.cfg.reward.type!r}; known: {sorted(REWARDS)}"
            )
        return REWARDS.get(self.cfg.reward.type)(self.task.answers)

    def trainer(self, policy: torch.nn.Module) -> OnlineRLTrainer:
        return OnlineRLTrainer(
            policy,
            self.tokenizer,
            self.cfg.variant,
            self.reward(),
            self.log_prob(),
            self.cfg.rollout,
            self.cfg.grpo,
            self.task.completion_length,
        )

    def evaluate(self, policy: torch.nn.Module, step: int) -> dict[str, float]:
        answers = self.task.answers
        seen = self.evaluator.accuracy(policy, self.task.seen_prompts, answers)
        unseen = self.evaluator.accuracy(policy, self.task.unseen_prompts, answers)
        n_seen, n_unseen = len(self.task.seen_prompts), len(self.task.unseen_prompts)
        total = (seen * n_seen + unseen * n_unseen) / (n_seen + n_unseen)
        return {"step": step, "seen": seen, "unseen": unseen, "all": total}

    # ------------------------------------------------------------------ the run
    def run(self, policy: torch.nn.Module, steps: int | None = None) -> RLHistory:
        """Train ``policy`` in place with online GRPO and record the history.

        Args:
            policy: the warm-started model (see :meth:`warm_start`).
            steps: override ``train.steps``.
        """
        torch.manual_seed(self.cfg.seed)
        rng = random.Random(self.cfg.seed)
        trainer = self.trainer(policy)
        estimator = trainer.log_prob
        tracked = self.task.unseen_prompts[: self.cfg.train.track_prompts]
        history = RLHistory(tracked_prompts=tracked, candidates=self.task.candidates)
        total = steps or self.cfg.train.steps

        def record(step: int) -> None:
            history.evals.append(self.evaluate(policy, step))
            history.distributions[step] = {
                p: estimator.answer_distribution(policy, p, self.task.candidates).tolist()
                for p in tracked
            }

        record(0)
        if tracked:  # what the warm-started policy samples for the first tracked prompt
            history.groups[0] = trainer.collect([tracked[0]]).group(0)
        pool = self.task.prompts
        for step in range(1, total + 1):
            prompts = rng.sample(pool, min(self.cfg.rollout.prompts_per_step, len(pool)))
            recording = step % self.cfg.train.eval_every == 0 or step == total
            if tracked and recording:  # put the tracked prompt in this batch so its group is logged
                prompts = [tracked[0], *[p for p in prompts if p != tracked[0]]][: len(prompts)]
            stats, batch = trainer.step(prompts)
            history.stats.append(stats)
            if recording:
                record(step)
                if tracked:
                    history.groups[step] = batch.group(0)
        logger.info("online RL done: {}", history.final_accuracy)
        return history
