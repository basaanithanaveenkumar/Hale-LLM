"""Reward section of an online RL config."""

from __future__ import annotations

from hale_llm.rl.rl_config_section import RLConfigSection


class RewardSection(RLConfigSection):
    """Which reward scores a completion (a key of :data:`REWARDS`)."""

    type: str = "exact_match"
