"""Explainer contract: train a toy model, trace its generation step by step, animate it.

Each paradigm in ``hale_llm.models`` has one explainer (one module per paradigm, registered at
the bottom of its file). An explainer

1. trains a tiny model of its variant on :data:`CORPUS` (:func:`train_toy`),
2. ``trace``-s generation from the first :attr:`Explainer.prompt_len` characters, recording what
   the sampler computes at every step (probabilities, times, reveals, drafts, loop passes ...).
   Traces mirror the registered samplers operation for operation, so with the same seed they
   produce the same tokens (``tests/unit/test_explainers.py`` checks this),
3. ``render``-s a short film: the shared intro (the data and its choice points), how training
   sees a sentence, generation in motion, and the outcome. All films share one visual grammar
   (see :mod:`~hale_llm.visualization.explainers.canvas`): sequence on top, the model box that
   pulses once per forward pass, outputs flying into place, a forward-pass counter.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, ClassVar

import torch
from hale_core.registry import NamedRegistry

from hale_llm.visualization.explainers.canvas import (
    BLUE,
    GOLD,
    GREEN,
    MUTED,
    ORANGE,
    RED,
    TEAL,
    GifWriter,
    Layout,
    Stage,
    ease,
)
from hale_llm.visualization.explainers.toy import ToyRun, train_toy

#: The sentences every toy model learns: 48 characters = 6 blocks of 8, two choice points
#: ("model" / "robot", "write" / "spell") so generation has real decisions to make.
CORPUS = (
    "the tiny model learned to write this sentence :)",
    "the tiny robot learned to write this sentence :)",
    "the tiny model learned to spell this sentence :)",
    "the tiny robot learned to spell this sentence :)",
)
SENTENCE = CORPUS[0]

#: variant name -> :class:`Explainer` subclass (one per paradigm in :mod:`hale_llm.models`).
EXPLAINERS = NamedRegistry("explainer", scope="hale_llm")


class Explainer(ABC):
    """Animated, step-by-step explanation of one generation paradigm.

    Subclasses set the class attributes and implement :meth:`trace` and :meth:`render`.

    Attributes:
        variant: registered variant name.
        title: headline of every frame.
        subtitle: one-line idea of the paradigm.
        train_kwargs: :func:`train_toy` keyword arguments (model overrides, steps ...).
        temperature: sampling temperature of the traced generation.
        sampling_steps: denoising / integration steps (diffusion-style samplers).
    """

    variant: ClassVar[str]
    title: ClassVar[str]
    subtitle: ClassVar[str]
    train_kwargs: ClassVar[dict[str, Any]] = {}
    texts: ClassVar[tuple[str, ...]] = CORPUS
    prompt_len: ClassVar[int] = 8
    temperature: ClassVar[float] = 0.7
    sampling_steps: ClassVar[int] = 16

    def train(self, seed: int = 0) -> ToyRun:
        """Fit a toy model of :attr:`variant` to :attr:`texts`."""
        return train_toy(self.variant, self.texts, seed=seed, **self.train_kwargs)

    def prompt(self, run: ToyRun) -> torch.Tensor:
        return run.ids[:, : self.prompt_len]

    @property
    def max_new_tokens(self) -> int:
        return len(self.texts[0]) - self.prompt_len

    @abstractmethod
    def trace(self, run: ToyRun, seed: int = 0) -> Any:
        """Instrumented generation; the result has an ``ids`` tensor ``(1, seq)``."""

    @abstractmethod
    def render(self, run: ToyRun, trace: Any, writer: GifWriter) -> None:
        """Append every frame of the animation to ``writer``."""

    def reference(self, run: ToyRun, seed: int = 0) -> torch.Tensor:
        """The registered sampler's output under the same seed (what :meth:`trace` must match)."""
        from hale_llm.core.registry import get_sampler

        torch.manual_seed(seed)
        return get_sampler(self.variant)(
            run.model, run.tokenizer, self.prompt(run), max_new_tokens=self.max_new_tokens,
            temperature=self.temperature, sampling_steps=self.sampling_steps,
        )

    def build(self, path: str | Path, run: ToyRun | None = None, seed: int = 0) -> Path:
        """Train (unless ``run`` is given), trace and render the GIF to ``path``."""
        run = run or self.train(seed)
        writer = GifWriter()
        self.render(run, self.trace(run, seed), writer)
        return writer.save(path)

    # ------------------------------------------------------------------ shared scenes
    def stage(self, phase: str, caption: str = "", color: str = TEAL) -> Stage:
        return Stage(self.title, self.subtitle, phase, caption, phase_color=color)

    def intro(self, run: ToyRun, writer: GifWriter) -> None:
        """The data: four sentences slide in, then their two choice points light up."""
        texts = run.texts
        branch = {i for i in range(len(texts[0])) if len({t[i] for t in texts}) > 1}
        layout = Layout(len(texts[0]), per_row=48, top=35.5, row_gap=5.4, tile_w=1.66, tile_h=4.3, gap=0.2)
        frames = 16
        for f in range(frames):
            s = self.stage("the data", "the toy model learns these 4 sentences")
            lit = ease((f - 9) / 4)
            for r, text in enumerate(texts):
                appear = ease((f + 2 - r * 2) / 3)
                for i, ch in enumerate(text):
                    x, y = layout.pos(i)
                    is_branch = i in branch
                    kind = "prompt" if i < self.prompt_len else ("new" if is_branch and lit > 0.5 else "clean")
                    s.tile(x, y - r * layout.row_gap + 2.0 * (1 - appear), ch, kind, w=layout.tile_w,
                           h=layout.tile_h, size=9.5, alpha=appear, glow=lit if is_branch else 0)
            s.text(50, 11.5, f"every sample starts from the prompt  '{texts[0][: self.prompt_len]}'", size=12,
                   color=BLUE, alpha=ease((f - 5) / 3))
            s.text(50, 6.5, "two real choices:   model or robot?    write or spell?", size=14, weight="bold",
                   color=GOLD, alpha=lit)
            writer.stage(s, 90)
        writer.hold(1900)

    def outro(self, run: ToyRun, writer: GifWriter, ids, headline: str, stats: list[tuple[str, str]],
              layout: Layout | None = None, color: str = ORANGE) -> None:
        """Final frame: the finished sentence, a verdict and a few numbers."""
        text = run.tokenizer.decode(ids)
        valid = text in run.texts
        layout = layout or Layout(len(text))
        for f in range(6):
            s = self.stage("result", headline, color)
            for i, ch in enumerate(text):
                x, y = layout.pos(i)
                s.tile(x, y, ch, "prompt" if i < self.prompt_len else "done", w=layout.tile_w, h=layout.tile_h,
                       glow=(1 - ease(f / 5)) * 0.8 if i >= self.prompt_len else 0)
            a = ease(f / 4)
            s.text(50, layout.bottom - 4.0, ("✓  one of the training sentences" if valid else
                                             "✗  not a training sentence: parallel choices mixed the two branches"),
                   size=13, weight="bold", color=GREEN if valid else RED, alpha=a)
            x0 = 50 - (len(stats) - 1) * 14
            for k, (value, label) in enumerate(stats):
                s.text(x0 + 28 * k, 12.0, value, size=24, weight="bold", color=GOLD, alpha=a)
                s.text(x0 + 28 * k, 6.8, label, size=10, color=MUTED, alpha=a)
            writer.stage(s, 70)
        writer.hold(3800)

    def verdict(self, run: ToyRun, ids) -> str:
        """Was the output one of the training sentences, or a mix of them?"""
        text = run.tokenizer.decode(ids)
        if text in run.texts:
            return f"'{text}' is a training sentence"
        return f"'{text}' mixes the branches"


def register_explainer(cls: type[Explainer]) -> type[Explainer]:
    """Register ``cls`` under its :attr:`Explainer.variant` (called at the bottom of each module).

    Args:
        cls: the explainer class.

    Returns:
        ``cls`` unchanged.

    Raises:
        hale_core.exceptions.RegistryError: if the variant already has an explainer.
    """
    return EXPLAINERS.add(cls.variant, cls)


def get_explainer(variant: str) -> Explainer:
    """A new explainer instance for ``variant``.

    Args:
        variant: registered variant name (``"diffusion"``, ``"looped"`` …).

    Raises:
        hale_core.exceptions.RegistryError: for a variant without an explainer (also a
            ``KeyError``), listing the registered ones.
    """
    return EXPLAINERS.get(variant)()
