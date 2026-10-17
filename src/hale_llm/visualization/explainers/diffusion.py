"""Masked diffusion (MDLM) explainer: absorbing-state corruption, parallel denoising t: 1 → 0."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from hale_llm.visualization.explainers.base import Explainer, register_explainer
from hale_llm.visualization.explainers.canvas import GifWriter
from hale_llm.visualization.explainers.masked import (
    Clock,
    MaskedStep,
    MaskedTrace,
    corruption_scene,
    render_masked_generation,
)
from hale_llm.visualization.explainers.toy import ToyRun


class DiffusionExplainer(Explainer):
    """Film of masked diffusion: the whole canvas denoised in parallel as ``t`` falls."""

    variant = "diffusion"
    title = "Masked Diffusion LM"
    subtitle = (
        "training: mask each character w.p. t, learn to fill the masks  ·  sampling: t runs 1 → 0"
    )
    train_kwargs = {"model": {"attn_type": "bidirectional", "use_time_cond": True}, "steps": 900}

    def trace(self, run: ToyRun, seed: int = 0) -> MaskedTrace:
        """Mirror of ``sample_diffusion`` recording predictions, confidences and reveals."""
        torch.manual_seed(seed)
        model, tok = run.model, run.tokenizer
        prompt = self.prompt(run)
        b, lp = prompt.shape
        seq, mask_id, steps = lp + self.max_new_tokens, tok.mask_token_id, self.sampling_steps
        x = torch.full((b, seq), mask_id, dtype=torch.long)
        x[:, :lp] = prompt
        out = MaskedTrace(ids=x)
        t_steps = torch.linspace(1.0, model.eps, steps + 1)
        with torch.no_grad():
            for i in range(steps):
                t_cur, t_next = t_steps[i], t_steps[i + 1]
                logits = model({"x_t": x, "t": t_cur.expand(b), "attention_mask": None})
                logits = logits / max(self.temperature, 1e-5)
                logits[..., mask_id] = -float("inf")
                probs = F.softmax(logits, dim=-1)
                pred = torch.multinomial(probs.reshape(-1, model.vocab_size), 1).reshape(b, seq)
                p_reveal = ((t_cur - t_next) / t_cur.clamp(min=1e-8)).clamp(0, 1)
                if i == steps - 1:
                    p_reveal = torch.ones_like(p_reveal)
                can = x.eq(mask_id)
                can[:, :lp] = False
                do = can & (torch.rand(b, seq) < p_reveal)
                x = torch.where(do, pred, x)
                out.steps.append(
                    MaskedStep(
                        float(t_cur),
                        float(t_next),
                        float(p_reveal),
                        pred[0].tolist(),
                        probs[0].gather(-1, pred[0, :, None]).squeeze(-1).tolist(),
                        set(do[0].nonzero().flatten().tolist()),
                        x[0].tolist(),
                    )
                )
        out.ids = x
        return out

    def render(self, run: ToyRun, trace: MaskedTrace, writer: GifWriter) -> None:
        """Draw the film into ``writer``: intro, how training sees a sentence, generation in motion,
        outcome.
        """
        self.intro(run, writer)
        model = run.model

        def predict(x_t: torch.Tensor, t: float) -> torch.Tensor:
            return F.softmax(model({"x_t": x_t, "t": torch.tensor([t])})[0], dim=-1)

        corruption_scene(
            self,
            run,
            writer,
            predict=predict,
            is_masked=lambda u, t: u < t,
            clock=Clock(lambda t: t, "share of characters masked = t"),
            sweep=(0.0, 0.5),
            train_t=0.5,
            sweep_caption="forward process: raise the noise t, each character is masked with probability t",
            train_caption="the denoiser guesses every masked character at once",
            loss_note="loss = cross-entropy on masked positions × 1/t",
        )
        render_masked_generation(
            self,
            run,
            trace,
            writer,
            clock=Clock(lambda t: 1 - t, "share of real characters", reverse=True),
            start_t=1.0,
            caption=lambda st: (
                f"t {st.t_from:.2f} → {st.t_to:.2f}: guess every ?, keep each with p = "
                f"{st.p_reveal:.0%}"
            ),
        )
        n = len(trace.steps)
        self.outro(
            run,
            writer,
            trace.ids[0],
            "denoised in parallel, in any order",
            [
                (f"{n}", "forward passes"),
                (f"{self.max_new_tokens}", "new characters"),
                (f"{self.max_new_tokens / n:.2f}", "characters / pass"),
            ],
        )


register_explainer(DiffusionExplainer)
