"""Film: how online RL moves the policy's answer distribution for one prompt, step by step."""

from __future__ import annotations

from collections.abc import Iterator, Sequence

from hale_llm.visualization.explainers.canvas import (
    CYAN,
    FAINT,
    GOLD,
    INK,
    MUTED,
    PANEL,
    PURPLE,
    TEAL,
    Stage,
    lerp,
)
from hale_llm.visualization.rl_films.rl_film import RLFilm


class PolicyEvolutionFilm(RLFilm):
    """Bars = ``π(answer | prompt)`` at each recorded RL step, morphing between snapshots.

    Left: the distribution over the answers ``00 … 18`` (everything else is summed into
    "other"), with the true answer in gold and, for self-reward runs, the initial majority in
    purple. Right: greedy accuracy on SFT-seen and unseen facts up to the current step. The data
    come from :class:`RLHistory` (``distributions`` and ``evals``), so the film is the run.

    Args:
        steps: recorded steps (sorted).
        distributions: one probability list per step (over ``candidates``).
        candidates: answer strings.
        answer: the true answer of this prompt.
        evals: ``{"step", "seen", "unseen"}`` accuracy records.
        prompt: the prompt string.
        title: film headline.
        highlight: an extra answer to mark (e.g. the initial majority vote).
        tweens: interpolated frames between snapshots.
    """

    SHOWN = [f"{n:02d}" for n in range(19)]

    def __init__(
        self,
        steps: Sequence[int],
        distributions: Sequence[Sequence[float]],
        candidates: Sequence[str],
        answer: str,
        evals: Sequence[dict],
        prompt: str,
        title: str = "The policy learns from its own samples",
        highlight: str | None = None,
        tweens: int = 3,
    ) -> None:
        if len(steps) != len(distributions) or not steps:
            raise ValueError("need one distribution per recorded step")
        self.steps = list(steps)
        index = {c: i for i, c in enumerate(candidates)}
        self.bars = []
        for dist in distributions:
            shown = [float(dist[index[c]]) for c in self.SHOWN]
            self.bars.append(shown + [max(0.0, 1.0 - sum(shown))])
        self.answer = answer
        self.evals = list(evals)
        self.prompt = prompt
        self.title = title
        self.highlight = highlight
        self.tweens = tweens

    @classmethod
    def from_history(cls, history, prompt: str, answer: str, **kwargs) -> PolicyEvolutionFilm:
        """Build from an :class:`RLHistory` for one of its tracked prompts."""
        steps = sorted(history.distributions)
        return cls(
            steps,
            [history.distributions[s][prompt] for s in steps],
            history.candidates,
            answer,
            history.evals,
            prompt,
            **kwargs,
        )

    def _draw(self, step: float, bars: Sequence[float]) -> Stage:
        p_true = bars[self.SHOWN.index(self.answer)] if self.answer in self.SHOWN else 0.0
        stage = Stage(
            self.title,
            f"π(answer | '{self.prompt}') during online GRPO",
            f"step {step:.0f}",
            f"p(correct = '{self.answer}') = {p_true:.0%}",
            phase_color=TEAL,
        )
        x0, y0, width, height = 4.0, 9.0, 58.0, 26.0
        labels = [*self.SHOWN, "other"]
        bw = width / len(labels)
        stage.box(x0 - 1.0, y0 - 4.5, x0 + width + 1.0, y0 + height + 2.0, face=PANEL, edge=FAINT)
        for i, (label, p) in enumerate(zip(labels, bars, strict=True)):
            colour = GOLD if label == self.answer else (PURPLE if label == self.highlight else CYAN)
            if label == "other":
                colour = MUTED
            x = x0 + i * bw
            h = max(0.0, min(1.0, p)) * height
            if h > 0.05:
                stage.box(
                    x + 0.25, y0, x + bw - 0.25, y0 + h, face=colour, edge="none", radius=0.25
                )
            stage.text(
                x + bw / 2,
                y0 - 1.6,
                label,
                size=6.8 if label != "other" else 6,
                color=INK if label in (self.answer, self.highlight) else MUTED,
            )
        stage.text(x0, y0 + height + 0.6, "1.0", size=7, color=MUTED, ha="left")
        # accuracy panel
        px0, py0, px1, py1 = 67.0, 9.0, 96.0, 35.0
        stage.box(px0 - 1.5, py0 - 4.5, px1 + 1.0, py1 + 2.0, face=PANEL, edge=FAINT)
        last = self.steps[-1] or 1
        for key, colour in (("seen", TEAL), ("unseen", GOLD)):
            pts = [(e["step"], e[key]) for e in self.evals if e["step"] <= step]
            if pts:
                xs = [lerp(px0, px1, s / last) for s, _ in pts]
                ys = [lerp(py0, py1, v) for _, v in pts]
                stage.polyline(xs, ys, color=colour, lw=2.2)
                stage.dot(xs[-1], ys[-1], r=0.45, color=colour)
                stage.text(
                    px1 - 0.5,
                    py0 + (2.5 if key == "seen" else 5.0),
                    f"{key}: {pts[-1][1]:.0%}",
                    size=8.5,
                    color=colour,
                    ha="right",
                )
        stage.text(
            (px0 + px1) / 2,
            py0 - 2.8,
            "greedy accuracy (SFT-seen / unseen facts)",
            size=8,
            color=MUTED,
        )
        legend = "gold = true answer" + (
            "  ·  purple = initial majority vote" if self.highlight else ""
        )
        stage.footer(legend)
        stage.progress(step / last)
        return stage

    def frames(self) -> Iterator[tuple[Stage, int]]:
        yield self._draw(self.steps[0], self.bars[0]), 1400
        for k in range(1, len(self.steps)):
            a, b = self.bars[k - 1], self.bars[k]
            for t in range(1, self.tweens + 1):
                u = t / self.tweens
                yield (
                    self._draw(
                        lerp(self.steps[k - 1], self.steps[k], u),
                        [lerp(x, y, u) for x, y in zip(a, b, strict=True)],
                    ),
                    70,
                )
        yield self._draw(self.steps[-1], self.bars[-1]), 2600
