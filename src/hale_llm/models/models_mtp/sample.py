"""Self-speculative MTP decoding: the extra heads draft, the main head verifies.

Every iteration runs **one** forward on ``prefix + drafts``. Walking the drafts left to right,
the main head's sample at each position is compared with the draft there: equal -> accepted, the
first mismatch is replaced by the main head's sample and the walk stops; if every draft holds,
the main head's sample after the last one is a bonus token. New drafts come from the extra heads
at the last verified position. Every emitted token is a sample from the main head conditioned
on the true prefix, so the output distribution is exactly autoregressive sampling of head 0 -
the heads only change how many tokens one forward produces (Stern et al., 2018,
arXiv:1811.03115; Gloeckle et al., 2024).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from hale_llm.core.registry import register_sampler
from hale_llm.exceptions import GenerationError


@dataclass
class SpeculativeStep:
    """One verify forward: the drafts it checked and what it emitted."""

    drafts: list[int]
    accepted: int  # leading drafts that matched the main head
    emitted: list[int] = field(default_factory=list)  # accepted drafts + correction / bonus


def _sample(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    if temperature <= 0:
        return logits.argmax(dim=-1)
    probs = F.softmax(logits / temperature, dim=-1)
    return torch.multinomial(probs, 1).squeeze(-1)


@register_sampler("mtp")
@torch.no_grad()
def sample_mtp(
    model,
    tokenizer,
    prompt_ids: torch.Tensor,
    max_new_tokens: int,
    temperature: float = 1.0,
    return_history: bool = False,
    **_,
):
    """Generate ``max_new_tokens`` tokens (batch size 1) with self-speculative decoding.

    ``model.speculative_trace`` holds a :class:`SpeculativeStep` per verify forward and
    ``model.tokens_per_forward`` the mean tokens emitted per forward.
    """
    if prompt_ids.size(0) != 1:
        raise GenerationError("speculative MTP decoding runs one sequence at a time")
    model.eval()
    pad_id = tokenizer.pad_token_id
    max_len = model.backbone.max_length

    def heads_at(seq: list[int]) -> tuple[list[torch.Tensor], int]:
        window = seq[-max_len:]
        ids = torch.tensor([window], device=prompt_ids.device)
        return model({"input_ids": ids, "attention_mask": (ids != pad_id).long()}), len(seq) - len(window)

    x = prompt_ids[0].tolist()
    target = len(x) + max_new_tokens
    history = [(0.0, prompt_ids.clone())] if return_history else None
    trace: list[SpeculativeStep] = []

    logits, _ = heads_at(x)
    first = int(_sample(logits[0][0, -1], temperature))
    drafts = [int(_sample(h[0, -1], temperature)) for h in logits[1:]]
    x.append(first)
    trace.append(SpeculativeStep(drafts=[], accepted=0, emitted=[first]))
    forwards = 1
    while len(x) < target:
        n = len(x)
        drafts = drafts[: max(0, min(len(drafts), max_len - n, target - n - 1))]
        logits, offset = heads_at(x + drafts)
        forwards += 1
        main = logits[0][0]
        emitted: list[int] = []
        accepted = 0
        last = n - 1 + len(drafts)
        for j, draft in enumerate(drafts):
            s = int(_sample(main[n - 1 + j - offset], temperature))
            if s != draft:
                emitted.append(s)
                last = n - 1 + j
                break
            emitted.append(draft)
            accepted += 1
        else:
            emitted.append(int(_sample(main[last - offset], temperature)))
        trace.append(SpeculativeStep(drafts=list(drafts), accepted=accepted, emitted=emitted))
        drafts = [int(_sample(h[0, last - offset], temperature)) for h in logits[1:]]
        x.extend(emitted)
        if return_history:
            done = min(len(x), target) - prompt_ids.size(1)
            history.append((done / max_new_tokens, torch.tensor([x[:target]], device=prompt_ids.device)))
    out = torch.tensor([x[:target]], device=prompt_ids.device)
    model.speculative_trace = trace
    model.tokens_per_forward = (target - prompt_ids.size(1)) / forwards
    return (out, history) if return_history else out
