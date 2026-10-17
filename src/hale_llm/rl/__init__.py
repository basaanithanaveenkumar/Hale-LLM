"""Online ("self") reinforcement learning for Hale LLMs: the policy learns from its own samples.

The loop (:class:`OnlineRLTrainer`) is GRPO. Sample G completions per prompt from the current
policy, score them (:data:`REWARDS`: a verifier, or a label-free self-reward), normalise rewards
within each group, and take clipped policy-gradient steps with a KL anchor. Model families plug
in through :data:`LOG_PROB_ESTIMATORS`: exact for autoregressive models, and the d1 one-step
estimate for masked diffusion. :class:`OnlineRLExperiment` builds a run from
``configs/rl/*.yaml``.

Exports are lazy (PEP 562), so importing the package does not import torch-heavy modules.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS = {
    "GRPOSection": "grpo_section",
    "RLConfigSection": "rl_config_section",
    "RewardSection": "reward_section",
    "RolloutSection": "rollout_section",
    "SFTSection": "sft_section",
    "TaskSection": "task_section",
    "TrainSection": "train_section",
    "ArithmeticTask": "tasks.arithmetic_task",
    "AutoregressiveLogProb": "autoregressive_log_prob",
    "ExactMatchReward": "exact_match_reward",
    "GRPOLossTerms": "grpo_loss_terms",
    "GRPOObjective": "grpo_objective",
    "GroupRelativeAdvantage": "group_relative_advantage",
    "LOG_PROB_ESTIMATORS": "rl_registry",
    "MajorityVoteReward": "majority_vote_reward",
    "MaskedDiffusionLogProb": "masked_diffusion_log_prob",
    "OnlineRLConfig": "online_rl_config",
    "OnlineRLError": "online_rl_error",
    "OnlineRLExperiment": "online_rl_experiment",
    "OnlineRLTrainer": "online_rl_trainer",
    "PolicyEvaluator": "policy_evaluator",
    "PolicyLogProb": "policy_log_prob",
    "REWARDS": "rl_registry",
    "RLHistory": "rl_history",
    "RLStepStats": "rl_step_stats",
    "RewardFunction": "reward_function",
    "RolloutBatch": "rollout_batch",
    "register_log_prob": "rl_registry",
    "register_reward": "rl_registry",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f"{__name__}.{_EXPORTS[name]}"), name)
    globals()[name] = value
    return value
