---
name: hale-llm-add-paradigm
description: Add a new language-modelling paradigm (or finish the MTP stub) in HALE-LLM as a models_<name> plugin with model, collate, loss, sampler and metrics, plus config and tests. Use when implementing a new generative objective (e.g. MTP heads, uniform-noise diffusion, speculative decoding) or changing an existing one.
---

# Adding a paradigm

1. `src/ha_llm/models/models_<name>/model.py`:

```python
@register_collate("<name>")
def collate(input_ids, tokenizer, **_):
    return {"input_ids": input_ids, "attention_mask": (input_ids != tokenizer.pad_token_id).long(), ...}

@register_variant("<name>")
class MyLM(nn.Module):
    def __init__(self, vocab_size, cfg):
        super().__init__()
        self.backbone = build_backbone(vocab_size, cfg.model)
    def forward(self, batch): ...

@register_loss("<name>")
def my_loss(model, batch) -> torch.Tensor: ...          # use ha_llm.losses.model_token_nll

@register_sampler("<name>")
@torch.no_grad()
def sample(model, tokenizer, prompt_ids, max_new_tokens, temperature=1.0, sampling_steps=None, return_history=False, **_): ...
```

   Optional: `metrics.py` with `@register_metric`. `loss.py` and `sample.py` re-export.
2. Add one import in `src/ha_llm/models/__init__.py`.
3. `variant` is a free string resolved through the registries, so no schema change is needed.
4. `configs/<name>/small.yaml` with `inherits: ../base.yaml`.
5. Tests: a shape test and an overfit sanity case in `tests/integration/test_overfit_sanity.py`
   (the loss must drop on the overfit text).
6. If the sampler returns `history`, the visualiser can render GIFs. Return
   `(x, [(t, x_t, meta), ...])`.

## Finishing MTP

`models_mtp/model.py` builds `n_mtp_heads - 1` extra heads but only returns and trains the
main head. To complete it: return `[main, *extra]` logits, add a loss term
`Σ_k CE(logits_k[:, :-k-1], input_ids[:, k+1:])`, and write a sampler that drafts k tokens
and verifies them with the main head.

## Invariants

- Diffusion-family losses weight masked-position CE by the ELBO factor (1/t for α_t = 1 − t)
  and normalise by the number of loss positions.
- Never let the model predict `[MASK]`: samplers set `logits[..., mask_id] = -inf`.
- Block diffusion requires `seq_len % block_size == 0`, and the backbone gets
  `block_size` through `build_backbone(..., block_size=...)`.
