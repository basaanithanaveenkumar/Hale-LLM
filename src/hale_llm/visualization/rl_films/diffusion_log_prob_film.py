"""Film: d1's one-pass log-probability estimate for a masked-diffusion policy."""

from __future__ import annotations

from collections.abc import Iterator, Sequence

import torch

from hale_llm.exceptions import VisualizationError
from hale_llm.visualization.explainers.canvas import CYAN, GOLD, MUTED, ORANGE, PURPLE, Stage
from hale_llm.visualization.rl_films.masked_view import MaskedView
from hale_llm.visualization.rl_films.rl_film import RLFilm


class DiffusionLogProbFilm(RLFilm):
    """Mask the whole completion, randomly mask some prompt tokens, run the model once, read logps.

    The views are computed from a real model with :meth:`from_model`, so the numbers on screen are
    the ones GRPO used: a different random prompt mask per inner epoch gives a slightly
    different estimate of the same rollout's log-probability.

    Args:
        prompt: prompt string.
        completion: the sampled completion.
        views: one :class:`MaskedView` per inner epoch.
    """

    def __init__(self, prompt: str, completion: str, views: Sequence[MaskedView]) -> None:
        if not views:
            raise VisualizationError("need at least one masked view")
        self.prompt = prompt
        self.completion = completion
        self.views = list(views)

    @classmethod
    @torch.no_grad()
    def from_model(
        cls,
        model,
        tokenizer,
        prompt: str,
        completion: str,
        prompt_mask_prob: float = 0.15,
        epochs: int = 3,
        seed: int = 0,
    ) -> DiffusionLogProbFilm:
        """Run the real estimator ``epochs`` times with recorded random prompt masks."""
        gen = torch.Generator().manual_seed(seed)
        p_ids = torch.tensor([tokenizer.encode(prompt)])
        c_ids = torch.tensor([tokenizer.encode(completion)])
        mask_id = tokenizer.mask_token_id
        views = []
        for _ in range(epochs):
            hide = torch.rand(p_ids.shape, generator=gen) < prompt_mask_prob
            x = torch.cat(
                [
                    torch.where(hide, torch.full_like(p_ids, mask_id), p_ids),
                    torch.full_like(c_ids, mask_id),
                ],
                dim=1,
            )
            logits = model({"x_t": x, "t": torch.ones(1), "attention_mask": None})[
                :, p_ids.shape[1] :
            ].float()
            logits[..., mask_id] = -float("inf")
            logp = logits.log_softmax(-1).gather(-1, c_ids[..., None])[0, :, 0]
            views.append(
                MaskedView(
                    tuple(int(i) for i in hide[0].nonzero().flatten()),
                    tuple(float(v) for v in logp),
                )
            )
        return cls(prompt, completion, views)

    def _x(self, i: int) -> float:
        n = len(self.prompt) + len(self.completion)
        return 50.0 - (n - 1) * 2.6 + i * 5.2

    def _draw(
        self,
        phase: str,
        caption: str,
        masked_completion: bool,
        view: MaskedView | None,
        show_logp: bool,
        pulse: float,
        epoch: int | None = None,
    ) -> Stage:
        stage = Stage(
            "Log-probability of a dLLM rollout (d1 / diffu-GRPO)",
            "π(y|x) sums over all unmasking orders → intractable; estimate it in one pass",
            phase,
            caption,
            phase_color=PURPLE,
        )
        y = 31.0
        hidden = set(view.hidden_prompt) if view else set()
        for i, ch in enumerate(self.prompt):
            masked = i in hidden
            stage.tile(
                self._x(i),
                y,
                "" if masked else ch,
                "mask" if masked else "prompt",
                label="prompt" if i == 0 else "",
            )
        for k, ch in enumerate(self.completion):
            x = self._x(len(self.prompt) + k)
            stage.tile(
                x,
                y,
                "" if masked_completion else ch,
                "mask" if masked_completion else "done",
                label="completion" if k == 0 else "",
            )
            if show_logp and view:
                stage.text(
                    x, y + 4.4, f"{view.token_logps[k]:.2f}", size=10, color=GOLD, weight="bold"
                )
                stage.text(x, y + 7.0, f"log p('{ch}')", size=7.5, color=MUTED)
        stage.model_box(
            "masked-diffusion LM",
            "one bidirectional pass at t = 1",
            pulse=pulse,
            box=(36.0, 9.0, 64.0, 20.0),
        )
        stage.arrow((50.0, y - 3.2), (50.0, 20.4), color=CYAN, alpha=0.4 + 0.6 * pulse)
        if show_logp and view:
            total = sum(view.token_logps)
            stage.counter(
                f"{total:.2f}",
                "log π̂(y | x) = Σ token log-probs",
                color=GOLD,
                xy=(84.0, 20.0),
                sub=f"inner epoch {epoch}" if epoch else "",
            )
        stage.footer(
            "orange = random prompt masking (p = 0.15): a new view per inner epoch "
            "regularises reusing one rollout"
        )
        if hidden:
            for i in hidden:
                stage.box(
                    self._x(i) - 1.9,
                    y - 2.6,
                    self._x(i) + 1.9,
                    y + 2.6,
                    face="none",
                    edge=ORANGE,
                    lw=2.0,
                    z=4,
                )
        return stage

    def frames(self) -> Iterator[tuple[Stage, int]]:
        yield (
            self._draw(
                "rollout", "a sampled rollout: prompt + completion", False, None, False, 0.0
            ),
            1500,
        )
        yield (
            self._draw("mask", "mask every completion token (as at t = 1)", True, None, False, 0.0),
            1500,
        )
        for e, view in enumerate(self.views, start=1):
            caption = (
                f"inner epoch {e}: also mask a random {len(view.hidden_prompt)} prompt token(s)"
            )
            yield self._draw("mask", caption, True, view, False, 0.0), 900
            for k in (0.4, 0.8, 1.0):
                yield self._draw("forward", caption, True, view, False, k), 80
            yield (
                self._draw(
                    "estimate",
                    "each token read independently (mean-field) and summed",
                    True,
                    view,
                    True,
                    0.4,
                    epoch=e,
                ),
                1800,
            )
        yield (
            self._draw(
                "estimate",
                "AR: exact log-prob from 1 causal pass; dLLM: 1-pass estimate",
                True,
                self.views[-1],
                True,
                0.0,
                epoch=len(self.views),
            ),
            2600,
        )
