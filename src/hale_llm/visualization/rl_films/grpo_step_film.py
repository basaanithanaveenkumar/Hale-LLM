"""Film: one GRPO step on one prompt: sample a group, score it, normalise, push probabilities."""

from __future__ import annotations

from collections.abc import Iterator, Sequence

from hale_llm.exceptions import VisualizationError
from hale_llm.visualization.explainers.canvas import (
    CYAN,
    FAINT,
    GOLD,
    GREEN,
    INK,
    MUTED,
    PURPLE,
    RED,
    TEAL,
    Stage,
    ease,
)
from hale_llm.visualization.rl_films.rl_film import RLFilm


class GRPOStepFilm(RLFilm):
    """Real rollouts of one prompt (from :class:`RLHistory.groups`) animated through a GRPO step.

    1. the policy samples ``G`` answers (they fly out of the model box);
    2. each answer is scored: ✓/✗ against the verifier, or agreement with the group majority
       when ``pseudo_label`` is given (label-free self-reward);
    3. advantages ``(r − mean) / std`` grow as bars around the group mean;
    4. the update raises the probability of positive-advantage answers and lowers the others.

    Args:
        prompt: the prompt string.
        completions: the ``G`` sampled answers.
        rewards: their rewards.
        advantages: their group-relative advantages.
        title: film headline.
        pseudo_label: majority answer when the reward is a self-reward (``None`` = verifier).
    """

    MODEL = (3.0, 12.0, 21.0, 30.0)

    def __init__(
        self,
        prompt: str,
        completions: Sequence[str],
        rewards: Sequence[float],
        advantages: Sequence[float],
        title: str = "One GRPO step",
        pseudo_label: str | None = None,
    ) -> None:
        if not (len(completions) == len(rewards) == len(advantages)):
            raise VisualizationError("completions, rewards and advantages must have the same length")
        self.prompt = prompt
        self.completions = list(completions)
        self.rewards = [float(r) for r in rewards]
        self.advantages = [float(a) for a in advantages]
        self.title = title
        self.pseudo_label = pseudo_label
        self.scale = max(1e-6, max(abs(a) for a in self.advantages))

    # ------------------------------------------------------------------ layout
    def _row_y(self, j: int) -> float:
        return 37.0 - j * (28.0 / max(1, len(self.completions) - 1))

    def _draw(
        self,
        phase: str,
        caption: str,
        shown: int,
        flying: tuple[int, float] | None = None,
        scored: int = 0,
        grow: float = 0.0,
        push: float = 0.0,
        pulse: float = 0.0,
    ) -> Stage:
        subtitle = (
            "reward = agreement with the group's own majority (no labels)"
            if self.pseudo_label
            else "reward = verifier (exact match with the true answer)"
        )
        stage = Stage(
            self.title,
            subtitle,
            phase,
            caption,
            phase_color={"sample": CYAN, "score": GOLD, "normalise": PURPLE, "update": GREEN}.get(
                phase, TEAL
            ),
        )
        stage.model_box("π_θ", "current policy", pulse=pulse, box=self.MODEL)
        for i, ch in enumerate(self.prompt):
            stage.tile(5.6 + i * 3.4, 34.5, ch, "prompt", scale=0.8)
        stage.text(12.0, 39.2, "prompt", size=8.5, color=MUTED)
        mean = sum(self.rewards) / len(self.rewards)
        if scored >= len(self.completions):
            stage.text(60.0, 40.5, f"group mean reward = {mean:.2f}", size=9.5, color=MUTED)
        zero_x = 60.0
        if grow > 0:
            stage.polyline([zero_x, zero_x], [6.5, 39.0], color=FAINT, lw=1.2)
            stage.text(zero_x, 5.0, "A = 0 (group mean)", size=8, color=MUTED)
        for j, answer in enumerate(self.completions):
            y = self._row_y(j)
            visible = j < shown
            if flying and flying[0] == j:
                t = ease(flying[1])
                x0, y0 = self.MODEL[2], (self.MODEL[1] + self.MODEL[3]) / 2
                x, yy = x0 + (31.0 - x0) * t, y0 + (y - y0) * t
                for k, ch in enumerate(answer):
                    stage.tile(x + k * 2.6, yy, ch, "new", scale=0.62)
                continue
            if not visible:
                continue
            majority = self.pseudo_label is not None and answer == self.pseudo_label and scored > j
            for k, ch in enumerate(answer):
                stage.tile(31.0 + k * 2.6, y, ch, "draft" if majority else "done", scale=0.62)
            if j < scored:
                ok = self.rewards[j] > 0.5
                stage.mark(38.6, y, ok, scale=0.85)
                stage.text(42.5, y, f"r={self.rewards[j]:.0f}", size=9, color=INK if ok else MUTED)
            if grow > 0:
                a = self.advantages[j]
                length = 14.0 * (a / self.scale) * grow
                colour = GREEN if a > 0 else (RED if a < 0 else MUTED)
                if abs(length) > 0.05:
                    x0, x1 = sorted((zero_x, zero_x + length))
                    stage.box(x0, y - 0.8, x1, y + 0.8, face=colour, edge="none", radius=0.4)
                stage.text(zero_x + 16.0, y, f"A = {a:+.2f}", size=8.5, color=colour, ha="left")
            if push > 0 and abs(self.advantages[j]) > 1e-6:
                a = self.advantages[j]
                colour = GREEN if a > 0 else RED
                stage.arrow(
                    (29.0, y),
                    (self.MODEL[2] + 0.5, (self.MODEL[1] + self.MODEL[3]) / 2),
                    color=colour,
                    lw=0.8 + 1.8 * min(1.0, abs(a) / self.scale),
                    alpha=push * 0.85,
                    rad=0.15 if y > 21 else -0.15,
                )
                stage.text(
                    92.0, y, "raise p" if a > 0 else "lower p", size=8.5, color=colour, alpha=push
                )
        if self.pseudo_label and scored >= len(self.completions):
            stage.text(
                31.0,
                4.0,
                f"majority answer = '{self.pseudo_label}' → pseudo-label",
                size=9,
                color=PURPLE,
                ha="left",
            )
        stage.footer("each row is one sampled answer; A = (r − mean) / std within the group")
        return stage

    def frames(self) -> Iterator[tuple[Stage, int]]:
        G = len(self.completions)
        yield self._draw("sample", f"The policy answers '{self.prompt}' {G} times", 0), 900
        for j in range(G):
            for step in (0.35, 0.7, 1.0):
                yield (
                    self._draw(
                        "sample",
                        "sampling at temperature > 1 (exploration)",
                        j,
                        flying=(j, step),
                        pulse=0.8,
                    ),
                    55,
                )
            yield (
                self._draw("sample", "sampling at temperature > 1 (exploration)", j + 1, pulse=0.3),
                120,
            )
        caption = (
            "score by agreement with the majority"
            if self.pseudo_label
            else "score every answer with the verifier"
        )
        for j in range(G):
            yield self._draw("score", caption, G, scored=j + 1), 230
        yield self._draw("score", caption, G, scored=G), 900
        for k in range(1, 9):
            yield (
                self._draw(
                    "normalise",
                    "advantage = how much better than the group average",
                    G,
                    scored=G,
                    grow=k / 8,
                ),
                70,
            )
        yield (
            self._draw(
                "normalise",
                "advantage = how much better than the group average",
                G,
                scored=G,
                grow=1.0,
            ),
            1300,
        )
        if all(abs(a) < 1e-6 for a in self.advantages):
            yield (
                self._draw(
                    "update",
                    "all rewards equal → all advantages 0 → no gradient",
                    G,
                    scored=G,
                    grow=1.0,
                ),
                2600,
            )
            return
        for k in range(1, 9):
            yield (
                self._draw(
                    "update",
                    "∇θ Σ A·log π(answer): push good answers up, bad ones down",
                    G,
                    scored=G,
                    grow=1.0,
                    push=k / 8,
                    pulse=k / 8,
                ),
                80,
            )
        yield (
            self._draw(
                "update",
                "∇θ Σ A·log π(answer): push good answers up, bad ones down",
                G,
                scored=G,
                grow=1.0,
                push=1.0,
                pulse=1.0,
            ),
            2600,
        )
