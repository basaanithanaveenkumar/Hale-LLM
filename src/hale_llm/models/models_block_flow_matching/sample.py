"""Block flow matching sampler: blocks left to right, Euler integration of the flow inside each.

For block ``b`` the earlier blocks are clean (prompt or already generated) and feed the clean
stream. The block starts fully masked at ``t = 0``; at every step the model predicts ``x_1`` for
the masked positions and each one is revealed with the exact path probability
``(kappa(t') - kappa(t)) / (1 - kappa(t))``. A final greedy pass fills anything still masked at
``t = 1``. ``history`` frames carry ``{"block", "num_blocks", "block_size"}`` like
``block_diffusion`` so the GIF renderer highlights the active block.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from hale_llm.core.registry import register_sampler


def _block_meta(block: int, num_blocks: int, block_size: int) -> dict:
    return {"block": block, "num_blocks": num_blocks, "block_size": block_size}


@register_sampler("block_flow_matching")
@torch.no_grad()
def sample_block_flow_matching(
    model,
    tokenizer,
    prompt_ids: torch.Tensor,
    max_new_tokens: int,
    temperature: float = 1.0,
    sampling_steps: int | None = None,
    return_history: bool = False,
    **_,
):
    """Generate ``max_length`` tokens (prompt + completion) block by block.

    ``max_new_tokens`` is bounded by the backbone length: the canvas is ``max_length`` tokens,
    like ``block_diffusion``; generation stops after the block holding the last new token.
    """
    model.eval()
    device = prompt_ids.device
    n = model.backbone.max_length
    block_size = model.block_size
    num_blocks = n // block_size
    steps = sampling_steps or model.steps_per_block
    bsz = prompt_ids.size(0)
    mask_id = tokenizer.mask_token_id
    path = model.path

    x = torch.full((bsz, n), mask_id, dtype=torch.long, device=device)
    lp = min(prompt_ids.size(1), n)
    x[:, :lp] = prompt_ids[:, :lp]
    is_prompt = torch.zeros(bsz, n, dtype=torch.bool, device=device)
    is_prompt[:, :lp] = True
    attention_mask = torch.ones(bsz, n, dtype=torch.long, device=device)
    last_block = min(num_blocks, -(-min(lp + max_new_tokens, n) // block_size))

    history = (
        [(0.0, x.clone(), _block_meta(-1, num_blocks, block_size))] if return_history else None
    )

    def predict(t_block: float, start: int, end: int) -> torch.Tensor:
        t = torch.ones(bsz, n, device=device)  # clean context sits at the end of the flow
        t[:, start:end] = t_block
        logits = model({"x_clean": x, "x_t": x, "t": t, "attention_mask": attention_mask})
        logits = logits[:, start:end, :]
        logits[..., mask_id] = -float("inf")
        return logits

    for b_idx in range(last_block):
        start, end = b_idx * block_size, (b_idx + 1) * block_size
        free = ~is_prompt[:, start:end]
        if not free.any():
            continue
        times = torch.linspace(0.0, 1.0, steps + 1).tolist()
        for t_cur, t_next in zip(times[:-1], times[1:], strict=True):
            logits = predict(t_cur, start, end) / max(temperature, 1e-5)
            sampled = torch.multinomial(
                F.softmax(logits, dim=-1).reshape(-1, model.vocab_size), 1
            ).reshape(bsz, block_size)
            block = x[:, start:end]
            p = path.unmask_probability(t_cur, t_next)
            reveal = block.eq(mask_id) & free & (torch.rand(bsz, block_size, device=device) < p)
            x[:, start:end] = torch.where(reveal, sampled, block)
            if return_history:
                history.append((t_next, x.clone(), _block_meta(b_idx, num_blocks, block_size)))

        still = x[:, start:end].eq(mask_id) & free
        if still.any():
            final = predict(1.0 - path.eps, start, end).argmax(dim=-1)
            x[:, start:end] = torch.where(still, final, x[:, start:end])
            if return_history:
                history.append((1.0, x.clone(), _block_meta(b_idx, num_blocks, block_size)))

    out = x[:, : last_block * block_size]
    return (out, history) if return_history else out
