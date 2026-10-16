"""Diagnostics of one online RL step."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RLStepStats:
    """What happened in one collect-and-update step.

    Attributes:
        step: 1-based step index.
        reward_mean: mean reward of the rollouts.
        zero_variance_fraction: groups with identical rewards (no gradient signal).
        unique_answers: mean number of distinct completions per group (exploration).
        policy_loss: clipped surrogate (last inner epoch).
        kl: KL to the reference policy (last inner epoch).
        clip_fraction: tokens whose ratio left the clip range.
    """

    step: int
    reward_mean: float
    zero_variance_fraction: float
    unique_answers: float
    policy_loss: float
    kl: float
    clip_fraction: float
