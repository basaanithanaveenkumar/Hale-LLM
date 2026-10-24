"""Draw a sentence and, under every character, what the model guesses comes next after pass t."""

from __future__ import annotations

from hale_llm.visualization.explainers.canvas import (
    FAINT,
    GREEN,
    MUTED,
    RED,
    Layout,
    Stage,
    glyph,
)
from hale_llm.visualization.looped_films.pass_trace import PassTrace

#: Sentence rows sit a little lower than in the explainer films to leave room for the guesses.
SENTENCE_TOP = 38.0


def sentence_layout(n: int) -> Layout:
    """Two rows of 24 tiles with extra room below each row for the guess line."""
    return Layout(n, per_row=24, top=SENTENCE_TOP, row_gap=9.0)


def lens_correct(trace: PassTrace, tokenizer, t: int) -> list[bool]:
    """Per position: is the pass-``t`` logit-lens top-1 the true next character?"""
    ids = trace.lens_ids[t][:, 0].tolist()
    return [tokenizer.token(ids[i]) == trace.text[i + 1] for i in range(len(trace.text) - 1)]


def draw_sentence(
    s: Stage,
    layout: Layout,
    trace: PassTrace,
    tokenizer,
    t: int | None,
    *,
    kinds: list[str] | None = None,
    guess_alpha: float = 1.0,
    tile_alpha: list[float] | None = None,
) -> None:
    """Sentence tiles plus, when ``t`` is given, the pass-``t`` next-character guess under each.

    Guesses are green when right, red when wrong; their opacity is the model's probability.
    """
    n = len(trace.text)
    for i, ch in enumerate(trace.text):
        x, y = layout.pos(i)
        s.tile(x, y, ch, kinds[i] if kinds else "clean", alpha=tile_alpha[i] if tile_alpha else 1.0)
        if t is None or i == n - 1:
            continue
        guess = tokenizer.token(int(trace.lens_ids[t][i, 0]))
        prob = float(trace.lens_probs[t][i, 0])
        ok = guess == trace.text[i + 1]
        s.box(
            x - 1.3,
            y - 4.9,
            x + 1.3,
            y - 2.7,
            face=FAINT,
            edge="none",
            radius=0.4,
            alpha=0.6 * guess_alpha,
        )
        s.text(
            x,
            y - 3.8,
            glyph(guess),
            size=9,
            family="DejaVu Sans Mono",
            weight="bold",
            color=GREEN if ok else RED,
            alpha=guess_alpha * (0.35 + 0.65 * prob),
        )
    s.text(
        layout.left - 0.8,
        layout.pos(0)[1] - 3.8,
        "guess →",
        size=7.5,
        color=MUTED,
        ha="right",
        alpha=guess_alpha if t is not None else 0.0,
    )
