"""Typed configuration of an online RL experiment, loaded from YAML with ``inherits:`` chains."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from loguru import logger
from pydantic import Field, model_validator

from hale_llm.rl.grpo_section import GRPOSection
from hale_llm.rl.online_rl_error import OnlineRLError
from hale_llm.rl.reward_section import RewardSection
from hale_llm.rl.rl_config_section import RLConfigSection
from hale_llm.rl.rollout_section import RolloutSection
from hale_llm.rl.sft_section import SFTSection
from hale_llm.rl.task_section import TaskSection
from hale_llm.rl.train_section import TrainSection


class OnlineRLConfig(RLConfigSection):
    """Everything an :class:`OnlineRLExperiment` needs.

    Example:
        >>> path = "configs/rl/arithmetic_autoregressive.yaml"
        >>> cfg = OnlineRLConfig.from_yaml(path)  # doctest: +SKIP
        >>> cfg.with_overrides({"rollout": {"temperature": 1.0}})  # doctest: +SKIP
    """

    variant: str = "autoregressive"
    seed: int = 0
    model: dict[str, Any] = Field(
        default_factory=lambda: {"d_model": 96, "n_layers": 3, "d_ff": 256}
    )
    task: TaskSection = Field(default_factory=TaskSection)
    sft: SFTSection = Field(default_factory=SFTSection)
    rollout: RolloutSection = Field(default_factory=RolloutSection)
    grpo: GRPOSection = Field(default_factory=GRPOSection)
    reward: RewardSection = Field(default_factory=RewardSection)
    train: TrainSection = Field(default_factory=TrainSection)

    @model_validator(mode="after")
    def _check(self) -> OnlineRLConfig:
        if self.variant not in ("autoregressive", "diffusion"):
            raise ValueError(
                f"online RL supports 'autoregressive' and 'diffusion', got {self.variant!r}"
            )
        return self

    # ------------------------------------------------------------------ loading
    @classmethod
    def from_yaml(cls, path: str | Path) -> OnlineRLConfig:
        """Load ``path``, resolving ``inherits:`` / ``_base_`` chains of any depth (child keys win).

        Inheritance is resolved by HaleBlocks'
        :class:`~hale_core.config.config_loader.ConfigLoader`.

        Raises:
            OnlineRLError: on a missing file or an inheritance cycle (the loader's
                :class:`~hale_core.exceptions.ConfigError` is chained as the cause).
        """
        from hale_core.config.config_loader import ConfigLoader
        from hale_core.exceptions import ConfigError

        try:
            raw = ConfigLoader().load(Path(path).resolve())
        except ConfigError as err:
            raise OnlineRLError(
                f"cannot load RL config {path} (missing file or inheritance cycle): {err}"
            ) from err
        logger.debug("rl config {}", path)
        return cls.model_validate(raw)

    def with_overrides(self, overrides: dict[str, Any]) -> OnlineRLConfig:
        """A copy with nested ``overrides`` deep-merged in and re-validated."""
        from hale_llm.config.merge import deep_merge

        return OnlineRLConfig.model_validate(deep_merge(self.model_dump(), overrides))
