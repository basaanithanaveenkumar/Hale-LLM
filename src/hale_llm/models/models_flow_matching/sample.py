"""Discrete flow-matching sampler: integrate the masking path from t = 0 (all masks) to 1."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from hale_llm.core.registry import register_sampler
from hale_llm.models.models_flow_matching.vector_field import mask_prob


@register_sampler("flow_matching")
@torch.no_grad()
def sample_flow_matching(
    model,
    tokenizer,
    prompt_ids,
    max_new_tokens,
    temperature=1.0,
    sampling_steps=32,
    return_history=False,
    **_,
):
    """Euler integration of the masking path from ``t = 0`` (all masks) to ``t = 1``.

    Each step unmasks a masked generated position with probability ``dt / mask_prob(t)``.

    Returns:
        ``(B, L_prompt + max_new_tokens)`` ids, or ``(ids, history)``.
    """
    model.eval()
    device = prompt_ids.device
    b, lp = prompt_ids.shape
    seq = lp + max_new_tokens
    mask_id = tokenizer.mask_token_id
    x = torch.full((b, seq), mask_id, device=device, dtype=torch.long)
    x[:, :lp] = prompt_ids
    dt = 1.0 / sampling_steps
    gen = torch.arange(seq, device=device).ge(lp).unsqueeze(0)
    history = [(0.0, x.clone())] if return_history else None
    for step in range(sampling_steps):
        t_val = step * dt
        t = torch.full((b,), t_val, device=device)
        logits = model({"x_t": x, "t": t}) / max(temperature, 1e-5)
        logits[..., mask_id] = -float("inf")
        pred = torch.multinomial(F.softmax(logits, dim=-1).reshape(-1, model.vocab_size), 1).reshape(b, seq)
        p_unmask = min(dt / max(mask_prob(t_val), model.eps), 1.0)
        do = x.eq(mask_id) & gen & (torch.rand(b, seq, device=device) < p_unmask)
        x = torch.where(do, pred, x)
        if return_history:
            history.append((t_val + dt, x.clone()))
    return (x, history) if return_history else x
