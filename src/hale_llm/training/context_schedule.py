"""Turn ``data.context_window`` plus ``train.context_stages`` into per-step plans."""

from __future__ import annotations

from loguru import logger

from hale_llm.config.schema import RunConfig
from hale_llm.training.context_stage_plan import ContextStagePlan


class ContextSchedule:
    """Which context window, RoPE base and batch size apply at each training step.

    Recipe followed (SmolLM3-style): pretrain at a modest context for 90-95% of the
    tokens, then extend to a much longer context for the last few percent, raising the
    RoPE base so the longer positions stay distinguishable and mixing long documents with
    normal data so short-context quality does not regress.

    Example for 1,000 steps, base 4096 / theta 10k, one stage 32768 / theta 1M at 0.95::

        steps   0..949 : context  4096, batch 32
        steps 950..999 : context 32768, batch 4   (same ~131k tokens per step)
    """

    def __init__(self, plans: list[ContextStagePlan]) -> None:
        if not plans:
            raise ValueError("a schedule needs at least the base stage")
        self._plans = plans

    @property
    def plans(self) -> list[ContextStagePlan]:
        return list(self._plans)

    @classmethod
    def from_config(cls, cfg: RunConfig, total_steps: int) -> ContextSchedule:
        """Validate the stages in ``cfg`` and resolve their start fractions into step numbers."""
        base_window = cfg.model.max_length
        base_batch = cfg.train.batch_size
        plans = [
            ContextStagePlan(
                name="pretrain",
                start_step=0,
                context_window=base_window,
                rope_theta=cfg.model.rope_theta,
                batch_size=base_batch,
                long_doc_fraction=cfg.data.long_doc_fraction,
                long_doc_min_tokens=cfg.data.long_doc_min_tokens,
            )
        ]
        stages = cfg.train.context_stages
        if stages:
            cls._require_extensible(cfg)

        tokens_per_batch = base_batch * base_window
        previous = plans[0]
        for stage in stages:
            start_step = int(stage.start_fraction * total_steps)
            if not 0.0 < stage.start_fraction < 1.0:
                raise ValueError(f"stage {stage.name!r}: start_fraction must be in (0, 1)")
            if start_step <= previous.start_step:
                raise ValueError(
                    f"stage {stage.name!r} starts at step {start_step}, not after the previous "
                    f"stage ({previous.start_step}); order stages by start_fraction"
                )
            if stage.context_window <= previous.context_window:
                raise ValueError(
                    f"stage {stage.name!r}: context_window {stage.context_window} must exceed "
                    f"the previous stage's {previous.context_window}"
                )
            if stage.rope_theta < previous.rope_theta:
                raise ValueError(f"stage {stage.name!r}: rope_theta must not decrease")
            plan = ContextStagePlan(
                name=stage.name,
                start_step=start_step,
                context_window=stage.context_window,
                rope_theta=stage.rope_theta,
                # Keep tokens per step about constant: longer windows, fewer of them.
                batch_size=max(1, tokens_per_batch // stage.context_window),
                long_doc_fraction=stage.long_doc_fraction,
                long_doc_min_tokens=stage.long_doc_min_tokens,
            )
            plans.append(plan)
            previous = plan

        for plan in plans:
            logger.info(
                "context stage '{}': from step {} | window {} | rope theta {:g} | batch {}",
                plan.name, plan.start_step, plan.context_window, plan.rope_theta, plan.batch_size,
            )
        return cls(plans)

    @staticmethod
    def _require_extensible(cfg: RunConfig) -> None:
        """Only a RoPE model can change its context length mid-run."""
        if cfg.model.pos_encoding != "rope":
            raise ValueError("context stages need model.pos_encoding: rope (a learned table has a fixed length)")

    def plan_at(self, step: int) -> ContextStagePlan:
        """The stage in force at ``step`` (the last one whose start is not after it)."""
        current = self._plans[0]
        for plan in self._plans:
            if plan.start_step <= step:
                current = plan
        return current
