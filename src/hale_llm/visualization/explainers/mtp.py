"""Multi-token prediction explainer: K heads per position, self-speculative draft → verify."""

from __future__ import annotations

import torch

from hale_llm.models.models_mtp.loss import head_losses
from hale_llm.models.models_mtp.sample import sample_mtp
from hale_llm.visualization.explainers.base import Explainer, register_explainer
from hale_llm.visualization.explainers.canvas import (
    CYAN,
    GOLD,
    GREEN,
    INK,
    MODEL_BOX,
    MUTED,
    ORANGE,
    PURPLE,
    RED,
    GifWriter,
    Layout,
    Stage,
    arc_path,
    ease,
    pop,
)
from hale_llm.visualization.explainers.toy import ToyRun
from hale_llm.visualization.explainers.traces.mtp_trace import MTPTrace  # noqa: F401 - re-export

HEADS = 4
#: Head rows of the training scene (y of each head's target row).
HEAD_ROWS = (27.5, 22.5, 17.5, 12.5)


class MTPExplainer(Explainer):
    """Film of self-speculative MTP decoding: draft with extra heads, verify with the main head."""
    variant = "mtp"
    title = "Multi-Token Prediction LM"
    subtitle = f"{HEADS} heads on one trunk predict the next {HEADS} characters  ·  heads draft ahead, head 0 verifies"
    train_kwargs = {"model": {"n_mtp_heads": HEADS}, "steps": 300}

    def trace(self, run: ToyRun, seed: int = 0) -> MTPTrace:
        """The registered self-speculative sampler, with its per-forward record."""
        torch.manual_seed(seed)
        ids = sample_mtp(run.model, run.tokenizer, self.prompt(run), self.max_new_tokens, self.temperature)
        return MTPTrace(ids, list(run.model.speculative_trace), run.model.tokens_per_forward)

    # ------------------------------------------------------------------ film
    def render(self, run: ToyRun, trace: MTPTrace, writer: GifWriter) -> None:
        """Draw the film into ``writer``: intro, how training sees a sentence, generation in motion, outcome."""
        self.intro(run, writer)
        self._training(run, writer)
        self._generation(run, trace, writer)
        n, new = len(trace.steps), self.max_new_tokens
        self.outro(run, writer, trace.ids[0], "drafted ahead, verified in the same forward pass",
                   [(f"{n}", "forward passes"), (f"{new}", "new characters"), (f"{new / n:.2f}", "characters / pass")])

    def _training(self, run: ToyRun, writer: GifWriter) -> None:
        """Head k's target row is the input shifted by k + 1; a sweep shows one column at a time."""
        tok = run.tokenizer
        batch = {"input_ids": run.ids, "attention_mask": torch.ones_like(run.ids), "labels": run.ids,
                 "pad_token_id": tok.pad_token_id}
        with torch.no_grad():
            losses = [float(v) for v in head_losses(run.model, batch)]
        text = run.text
        n = 22
        layout = Layout(n, per_row=n, top=34.0, tile_w=2.9, tile_h=3.6, gap=0.5)
        sweep = [(q, 3) for q in range(4)] + [(q, 1) for q in range(4, n)]
        for q, count in sweep:
            for _ in range(count):
                s = self.stage("training", f"position i learns the next {HEADS} characters, one head each")
                for i in range(n):
                    x, y = layout.pos(i)
                    s.tile(x, y, text[i], "clean" if i <= q else "ghost", w=2.9, h=3.6, size=11,
                           glow=0.7 if i == q else 0)
                    for k, hy in enumerate(HEAD_ROWS):
                        ch = text[i + k + 1]
                        if i < q:
                            s.tile(x, hy, ch, "done", w=2.9, h=3.6, size=11, alpha=0.75)
                        elif i == q:
                            s.tile(x, hy, ch, "new", w=2.9, h=3.6, size=11, glow=1.0)
                        else:
                            s.tile(x, hy, "", "slot", w=2.9, h=3.6)
                xq = layout.pos(q)[0]
                s.box(xq - 2.0, HEAD_ROWS[-1] - 2.5, xq + 2.0, layout.top + 2.5, face="none", edge=GOLD, lw=1.6,
                      radius=0.8, z=4)
                for k, hy in enumerate(HEAD_ROWS):
                    s.text(layout.left - 0.8, hy, f"head {k}", size=9, color=GOLD if k == 0 else PURPLE, ha="right")
                    s.text(layout.left + layout.row_width() + 0.8, hy, f"x+{k + 1}", size=9, color=MUTED, ha="left")
                s.text(layout.left - 0.8, layout.top, "input", size=9, color=MUTED, ha="right")
                s.text(50, 6.5, f"position {q} ('{text[q]}') → heads predict '{text[q + 1: q + HEADS + 1]}'  "
                       "→  the trunk must plan ahead", size=12, color=CYAN)
                writer.stage(s, 120 if q < 4 else 60)
        writer.hold(600)
        s = self.stage("training", "loss = mean of the 4 shifted cross-entropies")
        for k, (hy, loss) in enumerate(zip(HEAD_ROWS, losses, strict=True)):
            s.text(30, hy + 6.0, f"head {k}  (x+{k + 1})", size=12, color=GOLD if k == 0 else PURPLE, ha="right")
            s.bar(33, hy + 5.3, 38, 1.4, min(loss / max(losses), 1.0), color=GOLD if k == 0 else PURPLE)
            s.text(73, hy + 6.0, f"{loss:.3f}", size=12, color=INK, ha="left")
        s.text(50, 8.5, "further ahead = harder: head 0 is exact, the far heads must guess at the branch points",
               size=11.5, color=MUTED)
        s.text(50, 4.5, "(trained toy, loss per head on one sentence)", size=9.5, color=MUTED)
        writer.stage(s, 2600)

    # ------------------------------------------------------------------ generation
    def _frame(self, run: ToyRun, layout: Layout, committed: list[int], extra: list[tuple[str, str, dict]],
               caption: str, color: str, forwards: int, pulse: float, emitted: int, note: str = "") -> Stage:
        """Committed tokens, then ``extra`` tiles (drafts / verdicts), then empty slots."""
        tok = run.tokenizer
        s = self.stage("generation", caption, color)
        for i in range(layout.n):
            x, y = layout.pos(i)
            j = i - len(committed)
            if i < len(committed):
                s.tile(x, y, tok.token(committed[i]), "prompt" if i < self.prompt_len else "done")
            elif j < len(extra):
                char, kind, kw = extra[j]
                s.tile(x, y, char, kind, **kw)
            else:
                s.tile(x, y, "", "slot")
        s.model_box("Trunk + 4 heads", "drafts 3 characters ahead", pulse=pulse, color=PURPLE)
        s.arrow((12.5, layout.bottom - 0.6), (12.5, MODEL_BOX[3] + 0.4), color=PURPLE if pulse > 0.3 else MUTED,
                lw=2.0)
        if note:
            s.text(50, 15.0, note, size=26, weight="bold", color=GOLD)
        tpf = emitted / forwards if forwards else 0.0
        s.counter(str(forwards), sub=f"{tpf:.2f} characters / pass" if forwards else "")
        s.footer("purple = draft from heads 1-3   ·   green ✓ = head 0 agrees   ·   red ✗ = replaced by head 0")
        s.progress(emitted / self.max_new_tokens, PURPLE)
        return s

    def _generation(self, run: ToyRun, trace: MTPTrace, writer: GifWriter) -> None:
        tok = run.tokenizer
        x = self.prompt(run)[0].tolist()
        layout = Layout(len(trace.ids[0]))
        emitted = 0
        for k, step in enumerate(trace.steps):
            drafts = [tok.token(d) for d in step.drafts]
            draft_tiles = [(d, "draft", {"label": f"h{j + 1}", "label_color": PURPLE}) for j, d in enumerate(drafts)]
            slow = k < 3 or step.accepted < len(step.drafts)
            # 1. one forward reads committed + drafts
            for f in range(3 if slow else 2):
                u = (f + 1) / (3 if slow else 2)
                cap = (f"forward {k + 1}: check {len(drafts)} drafts and draft again, all in one pass" if drafts else
                       "forward 1: head 0 picks the next character, heads 1-3 draft the ones after")
                writer.stage(self._frame(run, layout, x, draft_tiles, cap, PURPLE, k + (f == 2 or not slow), ease(u),
                                         emitted), 110 if slow else 70)
            # 2. verify the drafts left to right
            verdict: list[tuple[str, str, dict]] = list(draft_tiles)
            for j in range(len(drafts)):
                ok = j < step.accepted
                if j > step.accepted:
                    break
                for f in range(2 if slow else 1):
                    verdict[j] = (drafts[j], "accept" if ok else "reject",
                                  {"label": "✓" if ok else "✗", "label_color": GREEN if ok else RED,
                                   "scale": pop((f + 1) / 2, 0.15)})
                    cap = (f"verify: head 0 agrees with draft {j + 1}" if ok else
                           f"verify: head 0 disagrees at draft {j + 1}  →  keep its own choice, drop the rest")
                    writer.stage(self._frame(run, layout, x, verdict, cap, GREEN if ok else ORANGE, k + 1, 0.0,
                                             emitted), 140 if slow else 70)
            # 3. the correction (or bonus) token from head 0, rejected drafts fall away
            tail = step.emitted[step.accepted:]
            final = [(tok.token(t), "accept", {}) for t in step.emitted[: step.accepted]]
            for f in range(3 if slow else 2):
                u = (f + 1) / (3 if slow else 2)
                tiles = list(final)
                for t in tail:
                    label = "bonus" if step.accepted == len(drafts) and drafts else ("head 0" if not drafts else "fix")
                    tiles.append((tok.token(t), "new", {"scale": pop(u, 0.25), "glow": 1 - u, "label": label,
                                                         "label_color": GOLD}))
                if step.accepted < len(drafts):
                    for d in drafts[step.accepted + 1:]:
                        tiles.append((d, "draft", {"alpha": 1 - ease(u)}))
                cap = f"+{len(step.emitted)} characters from forward {k + 1}"
                writer.stage(self._frame(run, layout, x, tiles, cap, GOLD, k + 1, 0.0, emitted + len(step.emitted),
                                         note=f"+{len(step.emitted)}"), 110 if slow else 70)
            if slow and step.accepted < len(step.drafts):
                writer.hold(600)
            x = x + list(step.emitted)
            emitted += len(step.emitted)
            # 4. the heads' new drafts fly out of the model for the next forward
            if k + 1 < len(trace.steps):
                nxt = [tok.token(d) for d in trace.steps[k + 1].drafts]
                for f in range(3 if slow else 2):
                    u = ease((f + 1) / (3 if slow else 2))
                    s = self._frame(run, layout, x, [], "heads 1-3 draft the next characters", PURPLE, k + 1, 0.0,
                                    emitted)
                    for j, d in enumerate(nxt):
                        target = layout.pos(len(x) + j)
                        px, py = arc_path((MODEL_BOX[2] - 2, 15.0 + 2 * j), target, u, lift=4)
                        s.tile(px, py, d, "draft", z=8)
                    writer.stage(s, 90 if slow else 60)
        writer.hold(600)


register_explainer(MTPExplainer)
