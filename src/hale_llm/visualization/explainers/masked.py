"""Shared scenes of the masked-token explainers (diffusion, flow matching and their block forms).

All four generate by filling a canvas of ``[MASK]`` tokens: at each step the denoiser guesses
*every* masked position at once and a schedule decides which guesses are kept. A
:class:`MaskedStep` records one such step; :func:`render_masked_generation` animates it the same
way for every variant (guesses flicker in, a few are kept and pop, the rest fall back to
``?``), so the four GIFs differ only where the paradigms do: time direction, schedule and block
structure. :func:`corruption_scene` animates the training corruption and the denoiser's guesses.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import torch

from hale_llm.visualization.explainers.canvas import (
    GOLD,
    GREEN,
    INK,
    MODEL_BOX,
    MUTED,
    ORANGE,
    RED,
    TEAL,
    GifWriter,
    Layout,
    Stage,
    ease,
    lerp,
    pop,
)
from hale_llm.visualization.explainers.toy import ToyRun

#: Mini-plot of the schedule, between the model box and the counter.
SCHEDULE_BOX = (27.0, 5.5, 60.0, 22.5)


@dataclass
class MaskedStep:
    """One denoising / integration step.

    Attributes:
        t_from / t_to: time before and after the step.
        p_reveal: probability that a masked position is revealed in this step.
        pred: the sampled prediction at every position.
        conf: the model's probability of that prediction at every position.
        revealed: positions revealed in this step.
        x: the canvas after the step.
        block: active block (block variants) or ``None``.
        final_fill: the step is the greedy clean-up pass at the end of a block.
    """

    t_from: float
    t_to: float
    p_reveal: float
    pred: list[int]
    conf: list[float]
    revealed: set[int]
    x: list[int]
    block: int | None = None
    final_fill: bool = False


@dataclass
class MaskedTrace:
    ids: torch.Tensor
    steps: list[MaskedStep] = field(default_factory=list)


@dataclass(frozen=True)
class Clock:
    """How a paradigm's time is drawn in the schedule mini-plot.

    Attributes:
        fn: plotted curve over ``t ∈ [0, 1]``.
        title: what the curve is.
        reverse: time runs 1 → 0 (diffusion) - the axis is drawn right to left so progress
            always moves left to right.
        extra: optional dashed comparison curve and its label.
    """

    fn: Callable[[float], float]
    title: str
    reverse: bool = False
    extra: tuple[Callable[[float], float], str] | None = None


# ---------------------------------------------------------------------------- training scene
def corruption_scene(
    explainer,
    run: ToyRun,
    writer: GifWriter,
    *,
    predict: Callable[[torch.Tensor, float], torch.Tensor],
    is_masked: Callable[[float, float], bool],
    clock: Clock,
    sweep: Sequence[float],
    train_t: float,
    sweep_caption: str,
    train_caption: str,
    loss_note: str,
    seed: int = 7,
) -> None:
    """Corrupt the sentence as time moves along ``sweep``, then show the denoiser's guesses.

    ``is_masked(u, t)`` decides, from one fixed uniform draw ``u`` per position (a coupling, so
    the animation is continuous), whether that position is masked at time ``t``. ``predict``
    returns per-position probabilities ``(seq, vocab)`` for a corrupted input at time ``t``.
    """
    tok = run.tokenizer
    text = run.text
    layout = Layout(len(text))
    g = torch.Generator().manual_seed(seed)
    u = torch.rand(len(text), generator=g).tolist()
    stops = [lerp(a, b, (f + 1) / 8) for a, b in zip(sweep[:-1], sweep[1:], strict=True) for f in range(8)]

    def masked_at(t: float) -> list[bool]:
        return [i >= explainer.prompt_len and is_masked(u[i], t) for i in range(len(text))]

    previous = masked_at(stops[0])
    for k, t in enumerate(stops):
        masked = masked_at(t)
        s = explainer.stage("training", sweep_caption, TEAL)
        for i, ch in enumerate(text):
            x, y = layout.pos(i)
            flipped = masked[i] != previous[i]
            if masked[i]:
                s.tile(x, y, "", "mask", glow=0.9 if flipped else 0)
            else:
                s.tile(x, y, ch, "prompt" if i < explainer.prompt_len else "clean", glow=0.9 if flipped else 0)
        s.schedule(SCHEDULE_BOX, clock.fn, t, title=clock.title, value=f"t = {t:.2f}", reverse=clock.reverse,
                   extra=clock.extra)
        s.counter(f"{sum(masked)}", label="masked characters", color=GOLD)
        s.footer("? = the [MASK] token  ·  one random draw per character decides when it gets masked")
        s.progress((k + 1) / len(stops), TEAL)
        writer.stage(s, 90)
        previous = masked
    writer.hold(700)

    masked = masked_at(train_t)
    x_t = torch.tensor([[tok.mask_token_id if m else tok.stoi[c] for c, m in zip(text, masked, strict=True)]])
    with torch.no_grad():
        guess = predict(x_t, train_t).argmax(-1).tolist()
    correct = sum(1 for i, m in enumerate(masked) if m and tok.token(guess[i]) == text[i])
    total = sum(masked)
    for f in range(10):
        a = ease((f - 3) / 4)
        s = explainer.stage("training", train_caption, TEAL)
        for i, ch in enumerate(text):
            x, y = layout.pos(i)
            if not masked[i]:
                s.tile(x, y, ch, "prompt" if i < explainer.prompt_len else "clean")
            elif a <= 0.01:
                s.tile(x, y, "", "mask")
            else:
                ok = tok.token(guess[i]) == ch
                s.tile(x, y, tok.token(guess[i]), "accept" if ok else "reject", scale=pop(a, 0.2) if f < 8 else 1.0)
                if not ok:
                    s.text(x, y - layout.tile_h / 2 - 1.2, f"→{ch}", size=8, color=RED, family="DejaVu Sans Mono",
                           alpha=a)
        s.model_box("Denoiser", "sees all positions", pulse=1 - abs(f - 2.5) / 3 if f < 6 else 0.0)
        s.text(44, 16.5, f"{correct} / {total}", size=26, weight="bold", color=GREEN, alpha=a)
        s.text(44, 11.0, "masked characters guessed right", size=10, color=MUTED, alpha=a)
        s.text(44, 7.2, loss_note, size=10, color=INK, alpha=a)
        s.counter(f"{train_t:.2f}", label="this example's t", color=GOLD)
        s.footer("green = right guess  ·  red = wrong (the loss pushes it toward the true character)")
        writer.stage(s, 110)
    writer.hold(2200)


# ---------------------------------------------------------------------------- generation scene
def render_masked_generation(
    explainer,
    run: ToyRun,
    trace: MaskedTrace,
    writer: GifWriter,
    *,
    clock: Clock,
    caption: Callable[[MaskedStep], str],
    model_name: str = "Denoiser",
    block_size: int | None = None,
    start_t: float,
) -> None:
    """Animate every step of ``trace``: guesses everywhere, a few kept, the rest re-masked."""
    tok = run.tokenizer
    mask_id = tok.mask_token_id
    n = len(trace.steps[0].x)
    layout = Layout(n, block=block_size)
    blocks = block_size is not None
    fast = blocks
    start = explainer.prompt(run)[0].tolist() + [mask_id] * (n - explainer.prompt_len)

    def frame(x: list[int], step: MaskedStep | None, k: int, phase: str, u: float, cap: str, color: str) -> Stage:
        s = explainer.stage("generation", cap, color)
        active = step.block if step is not None and blocks else None
        for i in range(n):
            px, py = layout.pos(i)
            blk = i // block_size if blocks else None
            in_scope = not blocks or blk == active
            future = blocks and active is not None and blk > active
            if i < explainer.prompt_len:
                s.tile(px, py, tok.token(x[i]), "prompt")
            elif x[i] != mask_id:
                fresh = step is not None and i in step.revealed and phase == "keep"
                s.tile(px, py, tok.token(x[i]), "new" if fresh else "done", scale=pop(u, 0.3) if fresh else 1.0,
                       glow=(1 - u) if fresh else 0)
            elif future:
                s.tile(px, py, "", "ghost")
            elif step is not None and in_scope and phase in ("guess", "keep"):
                a = ease(u) if phase == "guess" else 1 - ease(u)
                s.tile(px, py, "", "mask", ghost=tok.token(step.pred[i]), ghost_alpha=step.conf[i] * a)
            else:
                s.tile(px, py, "", "mask")
        if blocks and active is not None:
            x0, y0, x1, y1 = layout.block_span(active, pad=0.7)
            s.box(x0, y0, x1, y1, face="none", edge=ORANGE, lw=2.0, radius=0.9, z=5)
        pulse = (1 - 0.6 * ease(u)) if phase == "guess" else 0.0
        s.model_box(model_name, "fills every ? at once" if not blocks else "current block only", pulse=pulse)
        s.arrow((12.5, layout.bottom - 0.6), (12.5, MODEL_BOX[3] + 0.4), color=GOLD if pulse > 0.3 else MUTED, lw=2.0)
        if step is None:
            t = start_t
        else:
            t = step.t_from if phase == "guess" else lerp(step.t_from, step.t_to, ease(u))
        s.schedule(SCHEDULE_BOX, clock.fn, t, title=clock.title, value=f"t = {t:.2f}", reverse=clock.reverse,
                   extra=clock.extra)
        if step is not None and phase == "keep":
            s.text(72, 17.0, f"+{len(step.revealed)}", size=26, weight="bold", color=GOLD)
            s.text(72, 11.5, "kept this step", size=10, color=MUTED)
        left = sum(1 for i, v in enumerate(x) if v == mask_id and i >= explainer.prompt_len)
        s.text(72, 7.5, f"{left} ? left", size=10.5, color=INK)
        s.counter(str(k), sub="")
        s.footer("faint gold letter = the model's guess (brighter = more sure)  ·  only a few guesses are kept per step")
        s.progress(1 - left / (n - explainer.prompt_len), GOLD)
        return s

    cap0 = f"start: the prompt + {n - explainer.prompt_len} [MASK] tokens"
    writer.stage(frame(start, None, 0, "idle", 1.0, cap0, TEAL), 1600)
    x = start
    for k, step in enumerate(trace.steps):
        slow = not fast or k < 4
        cap = caption(step)
        for phase, count in (("guess", 3 if slow else 2), ("keep", 3 if slow else 2)):
            for f in range(count):
                shown = x if phase == "guess" else step.x
                writer.stage(frame(shown, step, k + 1, phase, (f + 1) / count, cap, GOLD if step.final_fill else ORANGE),
                             110 if slow else 75)
        x = step.x
        if blocks and (k + 1 == len(trace.steps) or trace.steps[k + 1].block != step.block):
            writer.hold(500)
    writer.hold(700)


__all__ = ["SCHEDULE_BOX", "Clock", "MaskedStep", "MaskedTrace", "corruption_scene", "render_masked_generation"]
