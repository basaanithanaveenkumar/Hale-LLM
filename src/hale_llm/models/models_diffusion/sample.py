"""Masked-diffusion sampler: start fully masked and reveal tokens as t runs from 1 to eps."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from hale_llm.core.registry import register_sampler


@register_sampler("diffusion")
@torch.no_grad()
def sample_diffusion(
    model,
    tokenizer,
    prompt_ids,
    max_new_tokens,
    temperature=1.0,
    sampling_steps=32,
    return_history=False,
    **_,
):
    """Generate by reverse diffusion from a fully masked canvas after the prompt.

    Each of ``sampling_steps`` steps samples every position and reveals each still-masked one with
    the schedule's reveal probability; the last step reveals everything.

    Returns:
        ``(B, L_prompt + max_new_tokens)`` ids, or ``(ids, history)`` with ``(t, ids)`` snapshots.
    """
    model.eval()
    device = prompt_ids.device
    b, lp = prompt_ids.shape
    seq = lp + max_new_tokens
    mask_id = tokenizer.mask_token_id
    x = torch.full((b, seq), mask_id, device=device, dtype=torch.long)
    x[:, :lp] = prompt_ids
    history = [(1.0, x.clone())] if return_history else None
    t_steps = torch.linspace(1.0, model.eps, sampling_steps + 1, device=device)
    for i in range(sampling_steps):
        t_cur, t_next = t_steps[i], t_steps[i + 1]
        t = t_cur.expand(b)
        logits = model({"x_t": x, "t": t, "attention_mask": None}) / max(temperature, 1e-5)
        logits[..., mask_id] = -float("inf")
        pred = torch.multinomial(F.softmax(logits, dim=-1).reshape(-1, model.vocab_size), 1).reshape(b, seq)
        p_reveal = model.schedule.reveal_probability(t_cur, t_next)
        if i == sampling_steps - 1:
            p_reveal = torch.ones_like(p_reveal)  # t_next = eps ~ 0: nothing stays masked
        can = x.eq(mask_id)
        can[:, :lp] = False
        do = can & (torch.rand(b, seq, device=device) < p_reveal)
        x = torch.where(do, pred, x)
        if return_history:
            history.append((float(t_next), x.clone()))
    return (x, history) if return_history else x
