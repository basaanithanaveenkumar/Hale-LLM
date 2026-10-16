"""Looped transformer explainer: one shared block iterated, ACT halting, logit lens per pass."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from hale_llm.visualization.explainers.base import Explainer, register_explainer
from hale_llm.visualization.explainers.canvas import (
    FAINT,
    GOLD,
    GREEN,
    INK,
    MUTED,
    PANEL,
    TEAL,
    GifWriter,
    Layout,
    bezier,
    ease,
    lerp,
)
from hale_llm.visualization.explainers.sequential import render_next_token
from hale_llm.visualization.explainers.toy import ToyRun

N_LOOPS = 6


@dataclass
class LoopPass:
    """State of the slow-motion forward after one pass of the shared block."""

    lens_ids: list[int]  # logit lens top-5 for the next token
    lens_probs: list[float]
    halting: list[float]  # cumulative ACT halting mass per token
    running: list[bool]
    change: list[float]  # |h_t - h_{t-1}| / |h_{t-1}| per token


@dataclass
class LoopGenStep:
    top_ids: list[int]
    top_probs: list[float]
    chosen: int
    loops: int  # passes the forward ran
    updates: float  # N(i) of the newest position


@dataclass
class LoopedTrace:
    ids: torch.Tensor
    passes: list[LoopPass] = field(default_factory=list)
    steps: list[LoopGenStep] = field(default_factory=list)


class LoopedExplainer(Explainer):
    variant = "looped"
    title = "Looped Transformer LM"
    subtitle = f"one weight-tied block applied up to {N_LOOPS}×  ·  ACT: every character decides when to stop"
    train_kwargs = {"model": {"arch": "looped", "n_layers": 1,
                              "loop": {"n_loops": N_LOOPS, "halting": "act", "act_init_bias": -1.0,
                                       "ponder_weight": 0.003}},
                    "steps": 500}

    def trace(self, run: ToyRun, seed: int = 0) -> LoopedTrace:
        out = LoopedTrace(ids=self.prompt(run))
        self._slow_motion(run, out)
        torch.manual_seed(seed)
        model, tok = run.model, run.tokenizer
        x = self.prompt(run)
        with torch.no_grad():  # mirror of sample_looped
            for _ in range(self.max_new_tokens):
                if x.size(1) > model.backbone.max_length:
                    x = x[:, -model.backbone.max_length :]
                logits = model({"input_ids": x, "attention_mask": (x != tok.pad_token_id).long()})
                stats = model.last_loop_stats
                probs = F.softmax(logits[:, -1, :] / max(self.temperature, 1e-5), dim=-1)
                nxt = torch.multinomial(probs, 1)
                top = probs[0].topk(5)
                updates = float(stats.token_updates[0, -1]) if stats.token_updates is not None else stats.loops
                out.steps.append(LoopGenStep(top.indices.tolist(), top.values.tolist(), int(nxt), stats.loops,
                                             updates))
                x = torch.cat([x, nxt], dim=1)
        out.ids = x
        return out

    def _slow_motion(self, run: ToyRun, out: LoopedTrace) -> None:
        """Capture the state after every pass of one forward on the prompt (logit lens + ACT)."""
        model = run.model
        stack, bb = model.backbone.stack, model.backbone
        states: list[torch.Tensor] = []
        hook = stack.core[-1].register_forward_hook(lambda _m, _i, o: states.append(o.detach()))
        try:
            with torch.no_grad():
                model({"input_ids": self.prompt(run)})
        finally:
            hook.remove()
        active = torch.ones(states[0].shape[:2], dtype=torch.bool)
        act = stack.halting.init_state(states[0], active)
        previous = None
        for t, state in enumerate(states):
            act = stack.halting.step(act, state, last=t + 1 == stack.spec.n_loops)
            lens = F.softmax(bb.lm_head(stack.ln_out(state))[0, -1], dim=-1).topk(5)
            change = ([1.0] * state.size(1) if previous is None else
                      ((state - previous).norm(dim=-1) / previous.norm(dim=-1).clamp_min(1e-6))[0].tolist())
            out.passes.append(LoopPass(lens.indices.tolist(), lens.values.tolist(), act.halting[0].tolist(),
                                       act.running[0].tolist(), change))
            previous = state

    # ------------------------------------------------------------------ film
    def render(self, run: ToyRun, trace: LoopedTrace, writer: GifWriter) -> None:
        self.intro(run, writer)
        self._loop(run, writer)
        self._slow_motion_scene(run, trace, writer)
        self._generation(run, trace, writer)
        n = len(trace.steps)
        mean = sum(st.updates for st in trace.steps) / n
        self.outro(run, writer, trace.ids[0], "one block of weights, as deep as each character needs",
                   [(f"{n}", "forward passes"), (f"{mean:.1f} / {N_LOOPS}", "loop passes per character"),
                    ("1", "block of parameters")])

    def _loop(self, run: ToyRun, writer: GifWriter) -> None:
        """One block W with a loop arrow; a state dot laps it while the unrolled strip fills."""
        core_params = sum(p.numel() for p in run.model.backbone.stack.core.parameters())
        w_box = (38.0, 18.0, 62.0, 31.0)
        exit_p, enter_p, ctrl = (58.0, w_box[1]), (42.0, w_box[1]), (50.0, 4.0)
        laps, per_lap = N_LOOPS, 6
        for lap in range(laps):
            for f in range(per_lap):
                u = (f + 1) / per_lap
                s = self.stage("the loop", "the SAME block runs again and again on its own output", GOLD)
                s.model_box("embed", "characters → vectors", box=(4.0, 20.0, 22.0, 29.0))
                s.model_box("shared block W", f"pass {lap + 1} of {N_LOOPS}", pulse=0.6 + 0.4 * (f == 0), color=GOLD,
                            box=w_box)
                s.model_box("LM head", "→ next character", box=(78.0, 20.0, 96.0, 29.0))
                s.arrow((22.4, 24.5), (w_box[0] - 0.4, 24.5), color=INK, lw=2.0)
                s.text(30.2, 26.3, "e", size=13, color=INK, style="italic")
                s.arrow((w_box[2] + 0.4, 24.5), (77.6, 24.5), color=MUTED, lw=2.0, alpha=0.5)
                pts = [bezier(exit_p, ctrl, enter_p, i / 30) for i in range(31)]
                s.polyline([p[0] for p in pts], [p[1] for p in pts], color=GOLD, lw=3.0, alpha=0.8)
                s.arrow(pts[-4], (enter_p[0], enter_p[1] + 0.2), color=GOLD, lw=3.0, head=14)
                s.dot(*bezier(exit_p, ctrl, enter_p, ease(u)), r=0.9, color=GOLD)
                s.text(50, 8.6, "h ← W(h + e)", size=12, color=GOLD, family="DejaVu Sans Mono")
                for k in range(N_LOOPS):
                    x0 = 22.0 + k * 9.6
                    lit = k < lap or (k == lap and u > 0.5)
                    s.box(x0, 35.0, x0 + 7.6, 39.0, face="#3A2A08" if lit else PANEL, edge=GOLD if lit else FAINT,
                          lw=1.4, radius=0.6)
                    s.text(x0 + 3.8, 37.0, "W", size=11, weight="bold", color=GOLD if lit else MUTED)
                    if k:
                        s.arrow((x0 - 2.0, 37.0), (x0 - 0.1, 37.0), color=GOLD if lit else FAINT, lw=1.2, head=7)
                s.text(20.5, 37.0, "unrolled:", size=9.5, color=MUTED, ha="right")
                s.counter(f"{lap + 1}×", label="compute depth", color=GOLD, xy=(88.0, 11.5))
                s.counter("1×", label=f"parameters ({core_params / 1e3:.0f}k)", color=TEAL, xy=(12.0, 11.5))
                s.progress((lap + u) / laps, GOLD)
                writer.stage(s, 80)
        writer.hold(1400)

    def _slow_motion_scene(self, run: ToyRun, trace: LoopedTrace, writer: GifWriter) -> None:
        """One forward on the prompt, pass by pass: halting rings fill, the logit lens sharpens."""
        tok = run.tokenizer
        prompt = self.prompt(run)[0].tolist()
        n = len(prompt)
        xs = [50 + (i - n / 2) * 8.0 for i in range(n + 1)]
        tile_y, ring_y = 33.0, 25.5
        frames = 6
        prev = [0.0] * n
        for t, p in enumerate(trace.passes):
            items = [(tok.token(i), q) for i, q in zip(p.lens_ids, p.lens_probs, strict=True) if q >= 0.03][:3]
            for f in range(frames):
                u = ease((f + 1) / frames)
                halted_now = sum(1 for r in p.running if not r)
                s = self.stage("one forward, slowed down",
                               f"after pass {t + 1}: {halted_now} of {n} characters have stopped (halted)", GOLD)
                for i, token_id in enumerate(prompt):
                    halted = not p.running[i] and u > 0.6
                    just_halted = halted and (t == 0 or trace.passes[t - 1].running[i])
                    s.tile(xs[i], tile_y, tok.token(token_id), "accept" if halted else "prompt", w=4.6, h=5.6,
                           size=16, glow=0.9 if just_halted else 0)
                    frac = lerp(prev[i], p.halting[i], u)
                    s.ring(xs[i], ring_y, 2.2, frac, color=GREEN if halted else GOLD)
                    s.text(xs[i], ring_y, f"{frac:.0%}" if frac < 0.995 else "✓", size=8.5 if frac < 0.995 else 12,
                           color=GREEN if halted else MUTED, weight="bold")
                top = items[0]
                s.tile(xs[n], tile_y, "", "slot", ghost=top[0], ghost_alpha=top[1], edge=GOLD, w=4.6, h=5.6, size=16)
                s.text(xs[n], tile_y + 4.3, "next?", size=9, color=GOLD)
                s.text(4.0, ring_y, "halting\nprobability", size=9, color=MUTED, ha="left")
                s.model_box("W", f"pass {t + 1} / {N_LOOPS}", pulse=1 - u, color=GOLD, box=(3.0, 6.0, 17.0, 17.0))
                s.candidates(28.0, 15.0, items, chosen=0, grow=u if t == 0 else 1.0,
                             title=f"if it stopped after pass {t + 1}, the next character would be")
                s.counter(f"{t + 1}", label="passes so far", color=GOLD)
                s.footer("ring = each character's halting probability (ACT); at ~100% it stops updating")
                writer.stage(s, 110)
            writer.hold(500)
            prev = p.halting
        writer.hold(1200)

    def _generation(self, run: ToyRun, trace: LoopedTrace, writer: GifWriter) -> None:
        steps = trace.steps

        def detail(k: int, phase: str, u: float) -> str:
            if phase == "read":
                return f"pass {max(1, round(u * steps[k].loops))} / {N_LOOPS}"
            return f"ran {steps[k].loops} / {N_LOOPS} passes"

        def passes_bar(s, layout: Layout, k: int, phase: str, u: float) -> None:
            """Under every generated character: how many loop passes its position used."""
            last = k if phase in ("read", "predict", "pick") else k + 1
            for j in range(last):
                x, y = layout.pos(self.prompt_len + j)
                used = steps[j].updates
                s.bar(x - layout.tile_w / 2, y - layout.tile_h / 2 - 1.1, layout.tile_w, 0.6, used / N_LOOPS,
                      color=GOLD if used > 2.5 else TEAL)

        render_next_token(
            self, run, trace.ids[0].tolist(), steps, writer,
            model_name="looped W", model_detail=detail, overlay=passes_bar,
            footer="bar under each new character = loop passes its position used before halting",
            counter_sub=lambda k: f"avg {sum(st.updates for st in steps[: k + 1]) / (k + 1):.1f} loops / char",
        )


register_explainer(LoopedExplainer)
