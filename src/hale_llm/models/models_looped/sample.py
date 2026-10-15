"""Looped LM decoding: left-to-right sampling; each new token gets the loop budget in force.

The budget comes from the caller - ``InferenceSession.generate(num_loops=...)``,
``sample.num_loops`` or ``hale-llm-sample --num-loops`` - or the ``num_loops`` argument here.
With ACT each token also stops on its own; with ``exit_tolerance`` the loop stops at its fixed
point. ``return_history`` records the passes used for every generated token.
"""

from __future__ import annotations

import contextlib

import torch
import torch.nn.functional as F
from hale_core.nn.stacks import loop_budget

from hale_llm.core.registry import register_sampler


@register_sampler("looped")
@torch.no_grad()
def sample_looped(
    model,
    tokenizer,
    prompt_ids: torch.Tensor,
    max_new_tokens: int,
    temperature: float = 1.0,
    return_history: bool = False,
    num_loops: int | None = None,
    **_,
):
    """Sample ``max_new_tokens`` tokens after ``prompt_ids``.

    Returns the ids, or ``(ids, history)`` where ``history`` holds ``(progress, ids)`` frames
    like every sampler; ``model.loops_per_token`` then lists the passes run for each token.
    """
    model.eval()
    x = prompt_ids
    pad_id = tokenizer.pad_token_id
    history = [(0.0, x.clone())] if return_history else None
    loops: list[int] = []
    budget = loop_budget(model, num_loops) if num_loops is not None else contextlib.nullcontext()
    with budget:
        for step in range(max_new_tokens):
            if x.size(1) > model.backbone.max_length:
                x = x[:, -model.backbone.max_length :]
            logits = model({"input_ids": x, "attention_mask": (x != pad_id).long()})
            loops.append(model.last_loop_stats.loops)
            next_logits = logits[:, -1, :] / max(temperature, 1e-5)
            nxt = torch.multinomial(F.softmax(next_logits, dim=-1), 1)
            x = torch.cat([x, nxt], dim=1)
            if return_history:
                history.append(((step + 1) / max_new_tokens, x.clone()))
    model.loops_per_token = loops
    return (x, history) if return_history else x
