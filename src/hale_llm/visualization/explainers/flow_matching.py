"""Discrete flow matching explainer: masking probability path, Euler integration t: 0 → 1."""

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


def _kappa(t: float) -> float:
    """Linear masking path: a character is real with probability κ(t) = t."""
    return t


class FlowMatchingExplainer(Explainer):
    """Film of discrete flow matching: tokens become real as ``t`` runs from 0 to 1."""
    variant = "flow_matching"
    title = "Discrete Flow Matching LM"
    subtitle = "a path from all masks (t = 0) to text (t = 1)  ·  sampling = Euler steps along it"
    train_kwargs = {"model": {"attn_type": "bidirectional", "use_time_cond": True}, "steps": 900}

    def trace(self, run: ToyRun, seed: int = 0) -> MaskedTrace:
        """Mirror of ``sample_flow_matching`` recording predictions, confidences and reveals."""
        torch.manual_seed(seed)
        model, tok = run.model, run.tokenizer
        prompt = self.prompt(run)
        b, lp = prompt.shape
        seq, mask_id, steps = lp + self.max_new_tokens, tok.mask_token_id, self.sampling_steps
        x = torch.full((b, seq), mask_id, dtype=torch.long)
        x[:, :lp] = prompt
        dt = 1.0 / steps
        gen = torch.arange(seq).ge(lp).unsqueeze(0)
        out = MaskedTrace(ids=x)
        with torch.no_grad():
            for step in range(steps):
                t_val = step * dt
                logits = model({"x_t": x, "t": torch.full((b,), t_val)}) / max(self.temperature, 1e-5)
                logits[..., mask_id] = -float("inf")
                probs = F.softmax(logits, dim=-1)
                pred = torch.multinomial(probs.reshape(-1, model.vocab_size), 1).reshape(b, seq)
                p_unmask = min(dt / max(1.0 - t_val, model.eps), 1.0)
                do = x.eq(mask_id) & gen & (torch.rand(b, seq) < p_unmask)
                x = torch.where(do, pred, x)
                out.steps.append(MaskedStep(
                    t_val, t_val + dt, p_unmask, pred[0].tolist(),
                    probs[0].gather(-1, pred[0, :, None]).squeeze(-1).tolist(),
                    set(do[0].nonzero().flatten().tolist()), x[0].tolist()))
        out.ids = x
        return out

    def render(self, run: ToyRun, trace: MaskedTrace, writer: GifWriter) -> None:
        """Draw the film into ``writer``: intro, how training sees a sentence, generation in motion, outcome."""
        self.intro(run, writer)
        model = run.model

        def predict(x_t: torch.Tensor, t: float) -> torch.Tensor:
            return F.softmax(model({"x_t": x_t, "t": torch.tensor([t])})[0], dim=-1)

        corruption_scene(
            self, run, writer, predict=predict, is_masked=lambda u, t: u >= _kappa(t),
            clock=Clock(_kappa, "κ(t) = share of real characters"), sweep=(0.0, 1.0, 0.45), train_t=0.45,
            sweep_caption="probability path: at time t each character is real with probability κ(t) = t",
            train_caption="training picks a t on the path and learns to fill the masks",
            loss_note="loss = cross-entropy on masked positions × κ′/(1 − κ)",
        )
        render_masked_generation(
            self, run, trace, writer, clock=Clock(_kappa, "κ(t) = share of real characters"), start_t=0.0,
            caption=lambda st: (f"Euler step t {st.t_from:.2f} → {st.t_to:.2f}: each ? jumps to its guess w.p. "
                                f"{st.p_reveal:.0%}"),
        )
        n = len(trace.steps)
        self.outro(run, writer, trace.ids[0], "integrated from all masks (t = 0) to text (t = 1)",
                   [(f"{n}", "forward passes"), (f"{self.max_new_tokens}", "new characters"),
                    (f"{self.max_new_tokens / n:.2f}", "characters / pass")])


register_explainer(FlowMatchingExplainer)
