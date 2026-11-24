from __future__ import annotations

import torch
import torch.nn.functional as F

from hale_llm.core.registry import register_sampler
from hale_llm.dataloader.padding import attention_mask_for


@register_sampler("autoregressive")
@torch.no_grad()
def sample_autoregressive(
    model,
    tokenizer,
    prompt_ids: torch.Tensor,
    max_new_tokens: int,
    temperature: float = 1.0,
    return_history: bool = False,
    **_,
):
    """Sample one token at a time; stop early once every sequence has produced EOS."""
    model.eval()
    x = prompt_ids
    pad_id = tokenizer.pad_token_id
    stop_id = tokenizer.eos_token_id
    finished = torch.zeros(x.size(0), dtype=torch.bool, device=x.device)
    history = [(0.0, x.clone())] if return_history else None
    for step in range(max_new_tokens):
        # Keep only the newest tokens when the text outgrows the context window.
        if x.size(1) > model.backbone.max_length:
            x = x[:, -model.backbone.max_length :]
        logits = model({"input_ids": x, "attention_mask": attention_mask_for(x, pad_id)})
        next_logits = logits[:, -1, :] / max(temperature, 1e-5)
        nxt = torch.multinomial(F.softmax(next_logits, dim=-1), 1)
        if stop_id is not None:
            # A row that already ended keeps emitting EOS, so the batch stays rectangular.
            nxt[finished] = stop_id
            finished |= nxt.squeeze(1) == stop_id
        x = torch.cat([x, nxt], dim=1)
        if return_history:
            history.append(((step + 1) / max_new_tokens, x.clone()))
        if stop_id is not None and bool(finished.all()):
            break
    return (x, history) if return_history else x
