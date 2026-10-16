"""Autoregressive LM explainer: teacher forcing (shifted targets), then read → predict → sample → append."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from hale_llm.visualization.explainers.base import Explainer, register_explainer
from hale_llm.visualization.explainers.canvas import (
    CYAN,
    GOLD,
    INK,
    MUTED,
    TEAL,
    GifWriter,
    Layout,
    ease,
)
from hale_llm.visualization.explainers.sequential import render_next_token
from hale_llm.visualization.explainers.toy import ToyRun


@dataclass
class ARStep:
    """One decoding step: the top-k next-token distribution, the sampled token, attention."""

    top_ids: list[int]
    top_probs: list[float]
    chosen: int
    chosen_prob: float
    attention: list[float] | None = None  # last layer, head-averaged, query = last position


@dataclass
class ARTrace:
    ids: torch.Tensor
    steps: list[ARStep] = field(default_factory=list)


def last_layer_attention(model, x: torch.Tensor) -> list[float] | None:
    """Head-averaged attention of the last position in the last layer (``None`` if unsupported)."""
    layers = getattr(model.backbone, "layers", None)
    if not layers:
        return None
    layer = layers[-1]
    mha = getattr(getattr(layer, "attn", None), "attn", None)
    if not isinstance(mha, torch.nn.MultiheadAttention):
        return None
    captured: list[torch.Tensor] = []
    hook = layer.register_forward_pre_hook(lambda _m, args: captured.append(args[0].detach()))
    try:
        model({"input_ids": x})
    finally:
        hook.remove()
    h = layer.ln1(captured[0])
    n = h.size(1)
    causal = torch.triu(torch.ones(n, n, dtype=torch.bool), 1)
    _, weights = mha(h, h, h, attn_mask=causal, need_weights=True, average_attn_weights=True)
    return weights[0, -1].tolist()


class AutoregressiveExplainer(Explainer):
    variant = "autoregressive"
    title = "Autoregressive LM"
    subtitle = "p(x) = Π p(xᵢ | x<ᵢ)  ·  causal attention  ·  one forward pass per new token"
    train_kwargs = {"steps": 300}

    def trace(self, run: ToyRun, seed: int = 0) -> ARTrace:
        """Mirror of ``sample_autoregressive`` that records each distribution and attention row."""
        torch.manual_seed(seed)
        model, tok = run.model, run.tokenizer
        x = self.prompt(run)
        out = ARTrace(ids=x)
        with torch.no_grad():
            for _ in range(self.max_new_tokens):
                if x.size(1) > model.backbone.max_length:
                    x = x[:, -model.backbone.max_length :]
                logits = model({"input_ids": x, "attention_mask": (x != tok.pad_token_id).long()})
                probs = F.softmax(logits[:, -1, :] / max(self.temperature, 1e-5), dim=-1)
                nxt = torch.multinomial(probs, 1)
                top = probs[0].topk(3)
                out.steps.append(ARStep(top.indices.tolist(), top.values.tolist(), int(nxt),
                                        float(probs[0, nxt]), last_layer_attention(model, x)))
                x = torch.cat([x, nxt], dim=1)
        out.ids = x
        return out

    # ------------------------------------------------------------------ film
    def render(self, run: ToyRun, trace: ARTrace, writer: GifWriter) -> None:
        self.intro(run, writer)
        self._training(run, writer)
        self._generation(run, trace, writer)
        n = len(trace.steps)
        self.outro(run, writer, trace.ids[0], "built left to right, one character per forward pass",
                   [(f"{n}", "forward passes"), (f"{n}", "new characters"), ("1.00", "characters / pass")])

    def _training(self, run: ToyRun, writer: GifWriter) -> None:
        """Teacher forcing: the target row is the input row shifted left by one."""
        text = run.text[:25]
        n = len(text) - 1
        layout = Layout(n, per_row=n, top=31.5)
        target_y = 19.5
        sweep = list(range(n))
        frames = [(q, 2) for q in sweep[:6]] + [(q, 1) for q in sweep[6:]]
        total = sum(c for _, c in frames)
        shown = 0
        for q, count in frames:
            for _ in range(count):
                shown += 1
                s = self.stage("training", "every position learns to predict the NEXT character")
                s.text(layout.left, 35.8, "input", size=10, color=MUTED, ha="left")
                s.text(layout.left, target_y - 3.4, "target = input shifted by one", size=10, color=MUTED,
                       ha="left")
                for i in range(n):
                    x, y = layout.pos(i)
                    visible = i <= q
                    s.tile(x, y, text[i], "clean" if visible else "ghost", glow=0.7 if i == q else 0)
                    if i < q:
                        s.tile(x, target_y, text[i + 1], "done")
                        s.arrow((x, y - 2.4), (x, target_y + 2.4), color=TEAL, lw=1.0, alpha=0.35, head=6)
                    elif i == q:
                        s.tile(x, target_y, text[i + 1], "new", glow=1.0)
                        s.arrow((x, y - 2.4), (x, target_y + 2.4), color=GOLD, lw=2.2, head=12)
                    else:
                        s.tile(x, target_y, "", "slot")
                x0 = layout.pos(0)[0] - layout.tile_w / 2
                xq = layout.pos(q)[0] + layout.tile_w / 2
                s.bar(x0, layout.top + 2.9, xq - x0, 0.45, 1.0, color=CYAN)
                s.text(50, 11.0, f"position {q} sees  '{text[: q + 1]}'  →  must predict  '{text[q + 1]}'",
                       size=12.5, color=CYAN, family="DejaVu Sans Mono")
                s.text(50, 6.8, "the causal mask hides everything to its right (the dim tiles)", size=10.5,
                       color=MUTED)
                s.progress(shown / total, TEAL)
                writer.stage(s, 110 if q < 6 else 60)
        s = self.stage("training", "one sentence = 47 next-character lessons, all in ONE forward pass")
        for i in range(n):
            x, y = layout.pos(i)
            s.tile(x, y, text[i], "clean")
            s.tile(x, target_y, text[i + 1], "done")
            s.arrow((x, y - 2.4), (x, target_y + 2.4), color=TEAL, lw=1.0, alpha=0.5, head=6)
        s.text(50, 9.5, "loss = average cross-entropy of every target  (the causal mask makes this parallel)",
               size=11.5, color=INK)
        s.progress(1.0, TEAL)
        writer.stage(s, 2400)

    def _generation(self, run: ToyRun, trace: ARTrace, writer: GifWriter) -> None:
        steps = trace.steps

        def attention(s, layout: Layout, k: int, phase: str, u: float) -> None:
            """Cyan underline under each context tile = attention weight of the new position."""
            att = steps[k].attention
            if not att or phase == "land":
                return
            a = ease(u) if phase == "read" else (1 - ease(u) if phase == "fly" else 1.0)
            top = max(att)
            for j, w in enumerate(att):
                x, y = layout.pos(j)
                frac = w / top
                if frac > 0.04:
                    s.box(x - layout.tile_w / 2, y - layout.tile_h / 2 - 1.05, x + layout.tile_w / 2,
                          y - layout.tile_h / 2 - 0.45, face=CYAN, edge="none", radius=0.25, alpha=a * frac)

        render_next_token(
            self, run, trace.ids[0].tolist(), steps, writer,
            model_name="Transformer", model_detail=lambda k, phase, u: "causal: sees only the left",
            overlay=attention, footer="cyan underline = where the new position looks (attention, last layer)",
            counter_sub=lambda k: "1 character / pass",
        )


register_explainer(AutoregressiveExplainer)
