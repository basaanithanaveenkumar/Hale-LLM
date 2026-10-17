"""Everything recorded during an online RL run (curves, answer distributions, example groups)."""

from __future__ import annotations

from dataclasses import dataclass, field

from hale_llm.rl.rl_step_stats import RLStepStats


@dataclass
class RLHistory:
    """Recorded by :class:`OnlineRLExperiment`, consumed by the notebook plots and films.

    Attributes:
        stats: one entry per RL step.
        evals: ``{"step", "seen", "unseen", "all"}`` greedy accuracies every ``eval_every`` steps
            (step 0 = after the warm start).
        tracked_prompts: prompts whose answer distribution is recorded.
        candidates: answers the distributions range over.
        distributions: ``step -> prompt -> probability per candidate``.
        groups: ``step -> (prompt, completions, rewards, advantages)`` of the first tracked prompt.
    """

    stats: list[RLStepStats] = field(default_factory=list)
    evals: list[dict[str, float]] = field(default_factory=list)
    tracked_prompts: list[str] = field(default_factory=list)
    candidates: list[str] = field(default_factory=list)
    distributions: dict[int, dict[str, list[float]]] = field(default_factory=dict)
    groups: dict[int, tuple[str, list[str], list[float], list[float]]] = field(default_factory=dict)

    def series(self, name: str) -> list[float]:
        """One :class:`RLStepStats` field over all steps (e.g. ``"reward_mean"``)."""
        return [getattr(s, name) for s in self.stats]

    @property
    def final_accuracy(self) -> dict[str, float]:
        """The last accuracy record (empty dict before any evaluation)."""
        return self.evals[-1] if self.evals else {}
