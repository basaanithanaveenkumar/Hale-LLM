"""Typed configuration of an online RL experiment, loaded from YAML with ``inherits:`` chains."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from loguru import logger
from pydantic import BaseModel, ConfigDict, Field, model_validator

from hale_llm.rl.online_rl_error import OnlineRLError


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid")


class TaskSection(_Section):
    """The toy environment: ``a+b=`` prompts with a two-digit answer."""

    name: Literal["arithmetic"] = "arithmetic"
    max_operand: int = Field(9, ge=1, le=9)
    sft_fraction: float = Field(0.4, gt=0.0, le=1.0)


class SFTSection(_Section):
    """Supervised warm start on a fraction of the facts (gives the policy a format and a prior)."""

    steps: int = Field(60, ge=0)
    lr: float = Field(3e-3, gt=0.0)


class RolloutSection(_Section):
    """How the policy samples its own training data."""

    group_size: int = Field(8, ge=2)
    prompts_per_step: int = Field(16, ge=1)
    temperature: float = Field(1.5, gt=0.0)
    sampling_steps: int = Field(2, ge=1)


class GRPOSection(_Section):
    """Group Relative Policy Optimisation hyper-parameters."""

    lr: float = Field(5e-4, gt=0.0)
    inner_epochs: int = Field(2, ge=1)
    clip_eps: float = Field(0.2, gt=0.0, lt=1.0)
    kl_coef: float = Field(0.02, ge=0.0)
    max_grad_norm: float = Field(1.0, gt=0.0)
    normalize_std: bool = True
    prompt_mask_prob: float = Field(0.15, ge=0.0, lt=1.0)


class RewardSection(_Section):
    """Which reward scores a completion (a key of :data:`REWARDS`)."""

    type: str = "exact_match"


class TrainSection(_Section):
    """Loop length and what is recorded for plots and animations."""

    steps: int = Field(1000, ge=1)
    eval_every: int = Field(50, ge=1)
    track_prompts: int = Field(3, ge=0)


class OnlineRLConfig(_Section):
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

        Inheritance is resolved by HaleBlocks' :class:`~hale_core.config.config_loader.ConfigLoader`.

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
