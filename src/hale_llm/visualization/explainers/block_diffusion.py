"""Block diffusion explainer: autoregressive over blocks, masked diffusion inside each block."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from hale_llm.visualization.explainers.base import Explainer, register_explainer
from hale_llm.visualization.explainers.blocks import two_stream_scene
from hale_llm.visualization.explainers.canvas import GifWriter, Layout
from hale_llm.visualization.explainers.masked import (
    Clock,
    MaskedStep,
    MaskedTrace,
    render_masked_generation,
)
from hale_llm.visualization.explainers.toy import ToyRun

BLOCK = 8


class BlockDiffusionExplainer(Explainer):
    variant = "block_diffusion"
    title = "Block Diffusion LM"
    subtitle = "blocks of 8, left to right (like AR)  ·  masked diffusion inside each block (t: 1 → 0)"
    train_kwargs = {"model": {"attn_type": "block_causal", "use_time_cond": True, "block_size": BLOCK},
                    "steps": 900}
    sampling_steps = 8

    def trace(self, run: ToyRun, seed: int = 0) -> MaskedTrace:
        """Mirror of ``sample_block_diffusion`` recording every within-block step."""
        torch.manual_seed(seed)
        model, tok = run.model, run.tokenizer
        prompt = self.prompt(run)
        n, bs, steps = model.backbone.max_length, model.block_size, self.sampling_steps
        bsz, mask_id = prompt.size(0), tok.mask_token_id
        x = torch.full((bsz, n), mask_id, dtype=torch.long)
        lp = min(prompt.size(1), n)
        x[:, :lp] = prompt[:, :lp]
        is_prompt = torch.zeros(bsz, n, dtype=torch.bool)
        is_prompt[:, :lp] = True
        attention_mask = torch.ones(bsz, n, dtype=torch.long)
        out = MaskedTrace(ids=x)

        def forward(t_noised: torch.Tensor) -> torch.Tensor:
            return model({"x_clean": x, "x_noised": x, "t_noised": t_noised, "attention_mask": attention_mask})

        with torch.no_grad():
            for b_idx in range(n // bs):
                start, end = b_idx * bs, (b_idx + 1) * bs
                if is_prompt[:, start:end].all():
                    continue
                t_steps = torch.linspace(1.0, 0.0, steps + 1)
                for step in range(steps):
                    t_cur, t_next = t_steps[step].item(), t_steps[step + 1].item()
                    t_noised = torch.zeros(bsz, n)
                    t_noised[:, start:end] = t_cur
                    logits = forward(t_noised)
                    logits[:, start:end, mask_id] = -float("inf")
                    probs = F.softmax(logits[:, start:end, :] / max(self.temperature, 1e-5), dim=-1)
                    sampled = torch.multinomial(probs.reshape(-1, model.vocab_size), 1).reshape(bsz, bs)
                    block_tokens = x[:, start:end]
                    p_unmask = 1.0 if t_cur <= 1e-8 else (t_cur - t_next) / t_cur
                    do = (block_tokens.eq(mask_id) & (torch.rand(bsz, bs) < p_unmask)
                          & ~is_prompt[:, start:end])
                    x[:, start:end] = torch.where(do, sampled, block_tokens)
                    pred = x[0].tolist()
                    pred[start:end] = sampled[0].tolist()
                    conf = [1.0] * n
                    conf[start:end] = probs[0].gather(-1, sampled[0, :, None]).squeeze(-1).tolist()
                    out.steps.append(MaskedStep(t_cur, t_next, p_unmask, pred, conf,
                                                {start + j for j in do[0].nonzero().flatten().tolist()},
                                                x[0].tolist(), block=b_idx))
                still = x[:, start:end].eq(mask_id) & ~is_prompt[:, start:end]
                if still.any():
                    logits = forward(torch.zeros(bsz, n))
                    logits[:, start:end, mask_id] = -float("inf")
                    final = logits[:, start:end, :].argmax(dim=-1)
                    x[:, start:end] = torch.where(still, final, x[:, start:end])
                    pred = x[0].tolist()
                    out.steps.append(MaskedStep(0.0, 0.0, 1.0, pred, [1.0] * n,
                                                {start + j for j in still[0].nonzero().flatten().tolist()},
                                                x[0].tolist(), block=b_idx, final_fill=True))
        out.ids = x
        return out

    def render(self, run: ToyRun, trace: MaskedTrace, writer: GifWriter) -> None:
        self.intro(run, writer)
        g = torch.Generator().manual_seed(3)
        times = (torch.rand(len(run.text) // BLOCK, generator=g) * 0.85 + 0.1).tolist()
        masked = [i >= self.prompt_len and bool(torch.rand((), generator=g) < times[i // BLOCK])
                  for i in range(len(run.text))]
        two_stream_scene(self, run, writer, block_size=BLOCK, times=times, masked=masked,
                         caption="training: a clean copy and a masked copy, one t per block")
        render_masked_generation(
            self, run, trace, writer, clock=Clock(lambda t: 1 - t, "block: share of real characters", reverse=True),
            block_size=BLOCK, start_t=1.0,
            caption=lambda st: (f"block {st.block}: t reached 0, the last ? take their most likely value"
                                if st.final_fill else
                                f"block {st.block} · t {st.t_from:.2f} → {st.t_to:.2f} · keep each guess w.p. "
                                f"{st.p_reveal:.0%}"),
        )
        n, blocks = len(trace.steps), len({st.block for st in trace.steps})
        self.outro(run, writer, trace.ids[0], "left to right over blocks, parallel inside each block",
                   [(f"{n}", "forward passes"), (f"{blocks} × {BLOCK}", "blocks × characters"),
                    (f"{self.max_new_tokens / n:.2f}", "characters / pass")], layout=Layout(len(run.text), block=BLOCK))


register_explainer(BlockDiffusionExplainer)
