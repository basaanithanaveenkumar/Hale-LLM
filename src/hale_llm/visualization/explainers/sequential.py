"""Shared generation scene of the one-token-per-forward explainers (autoregressive, looped).

Every new character is one cycle of the same loop, and the scene animates it literally:

1. **read** - the model box lights up while it reads the sequence so far,
2. **predict** - the next-character distribution grows out of it (real probabilities),
3. **pick** - one candidate is sampled (held longer at a genuine 50/50 decision),
4. **fly** - the sampled character flies into the next slot, which becomes context.

The first tokens and every decision point play in slow motion; confident stretches run at a
few frames per token so the GIF stays short. Paradigm-specific detail (attention, loop passes)
is drawn by an ``overlay`` callback on top of the shared frame.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Protocol

from hale_llm.visualization.explainers.canvas import (
    CYAN,
    GOLD,
    INK,
    MODEL_BOX,
    MUTED,
    ORANGE,
    GifWriter,
    Layout,
    Stage,
    arc_path,
    ease,
    pop,
)
from hale_llm.visualization.explainers.toy import ToyRun

#: Where the candidate list sits (stage units, centre of the first candidate tile).
CANDIDATES_XY = (29.5, 19.0)
#: A step is a "decision" when the top candidate has less than this probability.
DECISION = 0.9


class TokenStep(Protocol):
    """What the scene needs from one traced decoding step."""

    top_ids: list[int]
    top_probs: list[float]
    chosen: int


#: ``overlay(stage, layout, k, phase, u)`` draws paradigm-specific detail for step ``k``.
Overlay = Callable[[Stage, Layout, int, str, float], None]


def candidate_items(run: ToyRun, step: TokenStep, k: int = 3, floor: float = 0.03) -> tuple[list, int]:
    """Top candidates worth showing (``p >= floor``) and the index of the sampled one.

    The sampled token is always included, even if it was an unlikely draw.
    """
    tok = run.tokenizer
    pairs = [(i, p) for i, p in zip(step.top_ids, step.top_probs, strict=True) if p >= floor][:k]
    if step.chosen not in [i for i, _ in pairs]:
        prob = dict(zip(step.top_ids, step.top_probs, strict=True)).get(step.chosen, 0.0)
        pairs = pairs[: k - 1] + [(step.chosen, prob)]
    chosen = [i for i, _ in pairs].index(step.chosen)
    return [(tok.token(i), p) for i, p in pairs], chosen


def render_next_token(
    explainer,
    run: ToyRun,
    seq: Sequence[int],
    steps: Sequence[TokenStep],
    writer: GifWriter,
    *,
    model_name: str,
    model_detail: Callable[[int, str, float], str],
    overlay: Overlay | None = None,
    counter_sub: Callable[[int], str] | None = None,
    footer: str = "",
    slow_first: int = 3,
) -> None:
    """Animate ``steps`` decoding into ``seq`` (the final ids, prompt included)."""
    tok = run.tokenizer
    prompt_len = explainer.prompt_len
    layout = Layout(len(seq))
    n = len(steps)
    for k, step in enumerate(steps):
        pos = prompt_len + k
        items, chosen = candidate_items(run, step)
        decision = step.top_probs[0] < DECISION
        slow = decision or k < slow_first
        script = ([("read", 3), ("predict", 4), ("pick", 6 if decision else 2), ("fly", 4), ("land", 2)] if slow
                  else [("predict", 1), ("fly", 2)])
        slot = layout.pos(pos)
        cands = None
        for phase, count in script:
            for f in range(count):
                u = (f + 1) / count
                caption = ("a real choice: the training sentences branch here" if decision else
                           "one forward pass  →  one new character")
                s = explainer.stage("generation", caption, ORANGE if decision else CYAN)
                for i in range(len(seq)):
                    x, y = layout.pos(i)
                    if i < pos:
                        s.tile(x, y, tok.token(seq[i]), "prompt" if i < prompt_len else "done")
                    elif i == pos and phase == "land":
                        s.tile(x, y, tok.token(seq[i]), "new", scale=pop(u, 0.25), glow=1 - u)
                    elif i == pos:
                        s.tile(x, y, "", "slot", edge=GOLD, glow=0.5)
                    else:
                        s.tile(x, y, "", "slot")
                pulse = {"read": ease(u), "predict": 1 - 0.6 * ease(u)}.get(phase, 0.0)
                s.model_box(model_name, model_detail(k, phase, u), pulse=pulse)
                s.arrow((12.5, layout.bottom - 0.6), (12.5, MODEL_BOX[3] + 0.4), color=CYAN if pulse > 0.3 else MUTED,
                        lw=2.0)
                s.text(13.8, (layout.bottom + MODEL_BOX[3]) / 2, "reads everything so far", size=8.5,
                       color=MUTED, ha="left")
                if phase != "read":
                    grow = ease(u) if phase == "predict" else 1.0
                    picked = chosen if phase in ("pick", "fly", "land") or not slow else None
                    s.arrow((MODEL_BOX[2] + 0.4, 15.0), (CANDIDATES_XY[0] - 2.4, 15.0), color=MUTED, lw=1.6)
                    cands = s.candidates(*CANDIDATES_XY, items, chosen=picked, grow=grow,
                                         hide=chosen if phase in ("fly", "land") else None,
                                         alpha=1.0 if phase != "land" else 1 - 0.6 * u)
                    if decision and phase in ("pick", "fly", "land"):
                        top2 = "   vs   ".join(f"'{c}' {p:.0%}" for c, p in items[:2])
                        s.text(63, 19.5, top2, size=14, weight="bold", color=GOLD)
                        s.text(63, 15.0, f"sampled '{items[chosen][0]}'", size=11, color=INK)
                if phase == "fly" and cands is not None:
                    x, y = arc_path(cands[chosen], slot, ease(u), lift=6)
                    s.tile(x, y, items[chosen][0], "new", glow=1.0, z=8)
                if overlay is not None:
                    overlay(s, layout, k, phase, u)
                done = k + (0 if phase == "read" else 1)
                s.counter(str(done), sub=counter_sub(k) if counter_sub and phase != "read" else "")
                if footer:
                    s.footer(footer)
                s.progress((k + (u if phase in ("fly", "land") else 0)) / n, CYAN)
                writer.stage(s, 90 if slow else 60)
        if decision:
            writer.hold(700)
    writer.hold(500)


__all__ = ["CANDIDATES_XY", "DECISION", "Overlay", "TokenStep", "candidate_items", "render_next_token"]
