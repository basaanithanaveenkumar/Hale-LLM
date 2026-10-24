"""Inside the loop: after every pass, what would the model predict if it stopped right there?"""

from __future__ import annotations

from hale_llm.visualization.explainers.canvas import (
    CYAN,
    FAINT,
    GOLD,
    GREEN,
    INK,
    MUTED,
    PANEL,
    GifWriter,
    ease,
)
from hale_llm.visualization.looped_films.loop_film import LoopFilm, register_loop_film
from hale_llm.visualization.looped_films.loop_lab import LoopLab
from hale_llm.visualization.looped_films.sentence_panel import (
    draw_sentence,
    lens_correct,
    sentence_layout,
)

N_LOOPS = 6


class InsideAPassFilm(LoopFilm):
    """Logit lens per pass on a trained Universal-Transformer toy: guesses sharpen pass by pass.

    After pass ``t`` the hidden state of every position is pushed through the final norm and the
    LM head - the prediction the model *would* make if the loop stopped there. Early passes guess
    poorly, later passes converge on the sentence: depth is spent refining the same answer.
    """

    name = "inside_a_pass"
    title = "Inside the loop: every pass refines the guess"
    subtitle = (
        "logit lens - decode the hidden state after each pass of a trained 1-block, 6-pass model"
    )

    def render(self, lab: LoopLab, writer: GifWriter) -> None:
        """Draw the film into ``writer`` from a trained ``universal_transformer`` toy model."""
        run = lab.train("universal_transformer", n_loops=N_LOOPS)
        trace = lab.trace(run)
        tok = run.tokenizer
        layout = sentence_layout(len(trace.text))
        scores = [sum(lens_correct(trace, tok, t)) for t in range(trace.n_passes)]
        total = len(trace.text) - 1
        for t in range(trace.n_passes):
            for phase, count in (("run", 4), ("show", 5)):
                for g in range(count):
                    u = ease((g + 1) / count)
                    show = phase == "show"
                    caption = (
                        f"pass {t + 1}: the same block W runs on every position again"
                        if not show
                        else f"after pass {t + 1}: {scores[t]}/{total} next-character guesses right"
                    )
                    s = self.stage(
                        f"pass {t + 1} / {trace.n_passes}", caption, GOLD if show else CYAN
                    )
                    prev_t = t - 1 if t > 0 else None
                    draw_sentence(
                        s,
                        layout,
                        trace,
                        tok,
                        t if show else prev_t,
                        guess_alpha=u if show and t == 0 else 1.0,
                    )
                    s.model_box(
                        "shared block W",
                        f"pass {t + 1} of {trace.n_passes}",
                        pulse=(1 - u) if show else u,
                        color=GOLD,
                        box=(3.0, 4.5, 23.0, 15.5),
                    )
                    self._accuracy_plot(s, scores[: t + (1 if show else 0)], total)
                    s.progress((t + (0.5 if show else 0.0) + 0.5 * u) / trace.n_passes, GOLD)
                    s.footer(
                        "green = right guess for the next character, red = wrong · opacity = "
                        "confidence"
                    )
                    writer.stage(s, 85)
            writer.hold(500)
        self._closing(trace, tok, layout, scores, total, writer)

    def _closing(self, trace, tok, layout, scores, total, writer: GifWriter) -> None:
        """Name the remaining misses: they sit on the corpus's genuine choice points."""
        last = trace.n_passes - 1
        wrong = [i for i, ok in enumerate(lens_correct(trace, tok, last)) if not ok]
        caption = (
            f"{scores[last]}/{total} right - the misses are the data's real choices (model/robot, "
            f"write/spell)"
            if wrong
            else f"{scores[last]}/{total} right"
        )
        s = self.stage("result", caption, GOLD)
        draw_sentence(s, layout, trace, tok, last)
        for i in wrong:
            x, y = layout.pos(i)
            s.box(x - 1.9, y - 5.4, x + 1.9, y + 2.6, face="none", edge=GOLD, lw=2.0, radius=0.8)
        s.model_box(
            "shared block W", f"{trace.n_passes} passes", color=GOLD, box=(3.0, 4.5, 23.0, 15.5)
        )
        self._accuracy_plot(s, scores, total)
        s.footer("each pass costs one block application; the parameters are those of ONE block")
        writer.stage(s, 3600)

    @staticmethod
    def _accuracy_plot(s, scores: list[int], total: int) -> None:
        x0, y0, x1, y1 = 60.0, 4.0, 97.0, 16.5
        s.box(x0, y0, x1, y1, face=PANEL, edge=FAINT, radius=1.0)
        s.text(
            x0 + 1.4, y1 - 1.6, "right guesses after each pass", size=8.5, color=MUTED, ha="left"
        )
        px0, px1, py0, py1 = x0 + 3.0, x1 - 2.0, y0 + 1.6, y1 - 3.4
        s.polyline([px0, px1], [py0, py0], color=FAINT, lw=1.0)
        n = N_LOOPS
        pts = [
            (px0 + (px1 - px0) * k / (n - 1), py0 + (py1 - py0) * v / total)
            for k, v in enumerate(scores)
        ]
        if len(pts) > 1:
            s.polyline([p[0] for p in pts], [p[1] for p in pts], color=GREEN, lw=2.0)
        for px, py in pts:
            s.dot(px, py, r=0.45, color=GREEN)
        if scores:
            s.text(
                x1 - 1.4,
                y1 - 1.6,
                f"{scores[-1]}/{total}",
                size=10,
                color=INK,
                weight="bold",
                ha="right",
            )
        for k in range(n):
            s.text(px0 + (px1 - px0) * k / (n - 1), y0 + 0.6, str(k + 1), size=7, color=MUTED)


register_loop_film(InsideAPassFilm)
