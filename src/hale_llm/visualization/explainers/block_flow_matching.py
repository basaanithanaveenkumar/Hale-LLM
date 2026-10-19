"""Block flow matching explainer: autoregressive over blocks, a masking flow t: 0 → 1 in each."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from hale_llm.visualization.explainers.base import Explainer, register_explainer
from hale_llm.visualization.explainers.blocks import two_stream_scene
from hale_llm.visualization.explainers.canvas import MUTED, TEAL, GifWriter, Layout, ease
from hale_llm.visualization.explainers.masked import (
    Clock,
    MaskedStep,
    MaskedTrace,
    render_masked_generation,
)
from hale_llm.visualization.explainers.toy import ToyRun

BLOCK = 8


def _cosine_kappa(t: float) -> float:
    """Cosine masking path: a character is real with probability κ(t) = 1 − cos(πt/2)."""
    return 1.0 - math.cos(0.5 * math.pi * t)


def _linear(t: float) -> float:
    return t


class BlockFlowMatchingExplainer(Explainer):
    """Film of block flow matching: per-block flow from masks to data, blocks in order."""

    variant = "block_flow_matching"
    title = "Block Flow Matching LM"
    subtitle = (
        "blocks of 8, left to right  ·  inside each block a cosine flow from masks to text "
        "(t: 0 → 1)"
    )
    train_kwargs = {
        "model": {
            "attn_type": "block_causal",
            "use_time_cond": True,
            "block_size": BLOCK,
            "fm_schedule": "cosine",
        },
        "steps": 900,
    }
    sampling_steps = 8

    def trace(self, run: ToyRun, seed: int = 0) -> MaskedTrace:
        """Mirror of ``sample_block_flow_matching`` recording every Euler step."""
        torch.manual_seed(seed)
        model, tok = run.model, run.tokenizer
        prompt = self.prompt(run)
        n, bs, steps, path = (
            model.backbone.max_length,
            model.block_size,
            self.sampling_steps,
            model.path,
        )
        bsz, mask_id = prompt.size(0), tok.mask_token_id
        x = torch.full((bsz, n), mask_id, dtype=torch.long)
        lp = min(prompt.size(1), n)
        x[:, :lp] = prompt[:, :lp]
        is_prompt = torch.zeros(bsz, n, dtype=torch.bool)
        is_prompt[:, :lp] = True
        attention_mask = torch.ones(bsz, n, dtype=torch.long)
        last_block = min(n // bs, -(-min(lp + self.max_new_tokens, n) // bs))
        out = MaskedTrace(ids=x)

        def predict(t_block: float, start: int, end: int) -> torch.Tensor:
            t = torch.ones(bsz, n)
            t[:, start:end] = t_block
            logits = model({"x_clean": x, "x_t": x, "t": t, "attention_mask": attention_mask})[
                :, start:end, :
            ]
            logits[..., mask_id] = -float("inf")
            return logits

        with torch.no_grad():
            for b_idx in range(last_block):
                start, end = b_idx * bs, (b_idx + 1) * bs
                free = ~is_prompt[:, start:end]
                if not free.any():
                    continue
                times = torch.linspace(0.0, 1.0, steps + 1).tolist()
                for t_cur, t_next in zip(times[:-1], times[1:], strict=True):
                    probs = F.softmax(
                        predict(t_cur, start, end) / max(self.temperature, 1e-5), dim=-1
                    )
                    sampled = torch.multinomial(probs.reshape(-1, model.vocab_size), 1).reshape(
                        bsz, bs
                    )
                    block = x[:, start:end]
                    p = path.unmask_probability(t_cur, t_next)
                    reveal = block.eq(mask_id) & free & (torch.rand(bsz, bs) < p)
                    x[:, start:end] = torch.where(reveal, sampled, block)
                    pred = x[0].tolist()
                    pred[start:end] = sampled[0].tolist()
                    conf = [1.0] * n
                    conf[start:end] = probs[0].gather(-1, sampled[0, :, None]).squeeze(-1).tolist()
                    out.steps.append(
                        MaskedStep(
                            t_cur,
                            t_next,
                            p,
                            pred,
                            conf,
                            {start + j for j in reveal[0].nonzero().flatten().tolist()},
                            x[0].tolist(),
                            block=b_idx,
                        )
                    )
                still = x[:, start:end].eq(mask_id) & free
                if still.any():
                    final = predict(1.0 - path.eps, start, end).argmax(dim=-1)
                    x[:, start:end] = torch.where(still, final, x[:, start:end])
                    out.steps.append(
                        MaskedStep(
                            1.0,
                            1.0,
                            1.0,
                            x[0].tolist(),
                            [1.0] * n,
                            {start + j for j in still[0].nonzero().flatten().tolist()},
                            x[0].tolist(),
                            block=b_idx,
                            final_fill=True,
                        )
                    )
        out.ids = x[:, : last_block * bs]
        return out

    def render(self, run: ToyRun, trace: MaskedTrace, writer: GifWriter) -> None:
        """Draw the film into ``writer``: intro, how training sees a sentence, generation in motion,
        outcome.
        """
        self.intro(run, writer)
        g = torch.Generator().manual_seed(5)
        times = (torch.rand(len(run.text) // BLOCK, generator=g) * 0.9).tolist()
        masked = [
            i >= self.prompt_len
            and bool(torch.rand((), generator=g) >= _cosine_kappa(times[i // BLOCK]))
            for i in range(len(run.text))
        ]
        self._path(writer, times)
        two_stream_scene(
            self,
            run,
            writer,
            block_size=BLOCK,
            times=times,
            masked=masked,
            caption="training: a clean copy and a path sample, one t per block",
        )
        render_masked_generation(
            self,
            run,
            trace,
            writer,
            clock=Clock(_cosine_kappa, "block: κ(t) = share real", extra=(_linear, "κ = t")),
            block_size=BLOCK,
            start_t=0.0,
            caption=lambda st: (
                f"block {st.block}: t = 1, the last ? take their most likely value"
                if st.final_fill
                else f"block {st.block} · Euler t {st.t_from:.2f} → {st.t_to:.2f} · each ? jumps "
                "w.p. "
                f"{st.p_reveal:.0%}"
            ),
        )
        n, blocks = len(trace.steps), len({st.block for st in trace.steps})
        self.outro(
            run,
            writer,
            trace.ids[0],
            "left to right over blocks, a short flow inside each block",
            [
                (f"{n}", "forward passes"),
                (f"{blocks} × {BLOCK}", "blocks × characters"),
                (f"{self.max_new_tokens / n:.2f}", "characters / pass"),
            ],
            layout=Layout(len(run.text), block=BLOCK),
        )

    def _path(self, writer: GifWriter, times: list[float]) -> None:
        """The cosine path vs the linear one: few characters early, most of them near t = 1."""
        for f in range(24):
            t = ease((f + 1) / 18)
            s = self.stage(
                "the path", "cosine schedule: characters appear slowly, then all at once", TEAL
            )
            s.schedule(
                (18.0, 7.0, 82.0, 39.0),
                _cosine_kappa,
                t,
                title="κ(t) = 1 − cos(πt/2): share of real characters",
                value=f"t = {t:.2f}  ·  κ = {_cosine_kappa(t):.2f}",
                extra=(_linear, "linear κ = t"),
                dots=times if f >= 18 else (),
            )
            if f >= 18:
                s.text(
                    50,
                    3.0,
                    "orange dots = the t each training block drew in the next scene",
                    size=10,
                    color=MUTED,
                )
            writer.stage(s, 90)
        writer.hold(1600)


register_explainer(BlockFlowMatchingExplainer)
