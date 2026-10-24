"""``LoopLab``: train tiny looped models and look inside their loop.

Every looped film and notebook cell needs the same two things: a *trained* toy model of some
loop variant (so the pictures show learned behaviour, not random weights) and a pass-by-pass
record of one forward (:class:`PassTrace`). The lab provides both on the 4-sentence character
corpus of the explainer films, in a few seconds on a CPU, and caches trained models by variant.
"""

from __future__ import annotations

import math
import random
from typing import Any

import torch
import torch.nn.functional as F
from hale_core.nn.stacks import LoopedStack, loop_budget

from hale_llm.core.registry import get_loss, get_variant
from hale_llm.dataloader.collate import get_collate
from hale_llm.losses import with_auxiliary_losses
from hale_llm.visualization.explainers.base import CORPUS
from hale_llm.visualization.explainers.tokenizer import CharTokenizer
from hale_llm.visualization.explainers.toy import ToyRun, toy_config, train_toy
from hale_llm.visualization.looped_films.docstring_corpus import docstring_corpus
from hale_llm.visualization.looped_films.pass_trace import PassTrace


class LoopLab:
    """Trains and traces toy looped LMs (``variant: looped``) by loop preset.

    Args:
        texts: training sentences (equal length).
        steps: AdamW steps per toy-corpus model.
        seed: initialisation seed.
        text_steps: default AdamW steps of the docstring-corpus models (:meth:`train_text`).
    """

    def __init__(
        self,
        texts: tuple[str, ...] = CORPUS,
        steps: int = 500,
        seed: int = 0,
        text_steps: int = 1500,
    ) -> None:
        self.texts = texts
        self.steps = steps
        self.seed = seed
        self.text_steps = text_steps
        self._runs: dict[str, ToyRun] = {}

    def train(self, preset: str, n_layers: int = 1, **loop: Any) -> ToyRun:
        """A trained toy model of loop ``preset`` (with ``loop`` overrides), cached by its key."""
        key = f"{preset}|{n_layers}|{sorted(loop.items())}"
        if key not in self._runs:
            model = {"arch": "looped", "n_layers": n_layers, "loop": {"preset": preset, **loop}}
            self._runs[key] = train_toy(
                "looped", self.texts, model=model, steps=self.steps, seed=self.seed
            )
        return self._runs[key]

    # ------------------------------------------------------------------ real text
    def text_corpus(self) -> tuple[str, str]:
        """``(train, validation)`` split (90 / 10) of :func:`docstring_corpus`, cached."""
        if not hasattr(self, "_text"):
            text = docstring_corpus(alphabet="".join(CharTokenizer().chars))
            cut = int(0.9 * len(text))
            self._text = (text[:cut], text[cut:])
        return self._text

    def train_text(
        self,
        model: dict[str, Any],
        variant: str = "looped",
        steps: int | None = None,
        window: int = 64,
        batch_size: int = 64,
        lr: float = 3e-3,
    ) -> ToyRun:
        """A character LM of ``variant`` with ``model`` overrides, trained on the docstring corpus.

        Random ``window``-character crops form every batch, so the model has to generalise; use
        :meth:`text_loss` for held-out bits per character. Cached by its arguments.
        """
        steps = self.text_steps if steps is None else steps
        key = f"text|{variant}|{sorted(model.items(), key=str)}|{steps}|{window}"
        if key in self._runs:
            return self._runs[key]
        import hale_llm.models  # noqa: F401  - registers the variants

        train, _ = self.text_corpus()
        tok = CharTokenizer()
        torch.manual_seed(self.seed)
        cfg = toy_config(variant, "x" * window, model)
        net = get_variant(variant)(vocab_size=tok.vocab_size, cfg=cfg)
        loss_fn, collate = get_loss(variant), get_collate(variant)
        opt = torch.optim.AdamW(net.parameters(), lr=lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
        rng = random.Random(self.seed + 1)
        losses: list[float] = []
        net.train()
        for _ in range(steps):
            rows = [
                tok.encode(train[i : i + window])
                for i in (rng.randrange(len(train) - window) for _ in range(batch_size))
            ]
            task = loss_fn(net, collate(torch.tensor(rows), tok, cfg=cfg))
            loss, _ = with_auxiliary_losses(net, task)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            sched.step()
            losses.append(float(task.detach()))
        net.eval()
        run = ToyRun(variant, net, tok, cfg, [train[:window]], losses)
        self._runs[key] = run
        return run

    @torch.no_grad()
    def text_loss(
        self, run: ToyRun, n_loops: int | None = None, batches: int = 20, window: int = 64
    ) -> float:
        """Held-out bits per character of ``run`` (optionally at a test-time loop budget)."""
        _, val = self.text_corpus()
        tok, model = run.tokenizer, run.model
        loss_fn, collate = get_loss(run.variant), get_collate(run.variant)
        rng = random.Random(1234)
        total = 0.0
        with loop_budget(model, n_loops):
            for _ in range(batches):
                rows = [
                    tok.encode(val[i : i + window])
                    for i in (rng.randrange(len(val) - window) for _ in range(64))
                ]
                total += float(loss_fn(model, collate(torch.tensor(rows), tok, cfg=run.cfg)))
        return total / batches / math.log(2)

    @staticmethod
    def stack(run: ToyRun) -> LoopedStack:
        """The :class:`LoopedStack` of a toy run."""
        return run.model.backbone.stack

    @torch.no_grad()
    def trace(self, run: ToyRun, text: str | None = None, top_k: int = 3) -> PassTrace:
        """Run ``text`` (default: the first sentence) once and record every pass."""
        text = run.text if text is None else text
        model, stack = run.model, self.stack(run)
        ids = torch.tensor([run.tokenizer.encode(text)])
        with stack.record_passes() as states:
            model({"input_ids": ids})
        n_scratch = stack.spec.scratch_tokens
        out = PassTrace(text=text, loops=stack.last_stats.loops)
        previous = None
        act = None
        if stack.halting is not None:
            act = stack.halting.init_state(
                states[0], torch.ones(states[0].shape[:2], dtype=torch.bool)
            )
        for t, state in enumerate(states):
            h = state[0, n_scratch:]
            out.states.append(h)
            lens = F.softmax(model.backbone.lm_head(stack.ln_out(h)), dim=-1).topk(top_k, dim=-1)
            out.lens_ids.append(lens.indices)
            out.lens_probs.append(lens.values)
            if previous is None:
                out.change.append(torch.ones(h.shape[0]))
            else:
                out.change.append(
                    (h - previous).norm(dim=-1) / previous.norm(dim=-1).clamp_min(1e-6)
                )
            previous = h
            if act is not None:
                act = stack.halting.step(act, state, last=t + 1 == len(states))
                out.halting.append(act.halting[0, n_scratch:].clone())
                out.running.append(act.running[0, n_scratch:].clone())
        stats = stack.last_stats
        if stats.token_updates is not None:
            out.depth = stats.token_updates[0].round().long()
            if stack.router is not None:
                out.running = [out.depth > t for t in range(len(states))]
        return out
