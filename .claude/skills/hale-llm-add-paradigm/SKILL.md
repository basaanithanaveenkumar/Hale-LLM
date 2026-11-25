---
name: hale-llm-add-paradigm
description: Add a new model variant to HALE-LLM as a models_<name> package with model, loss, sampler and collate, plus config and tests. Use when adding a new architecture or objective on top of the shared transformer blocks.
---

# Add a model variant

1. Create `src/hale_llm/models/models_<name>/` with:
   - `model.py`: an `nn.Module` decorated with `@register_variant("<name>")`, constructed as
     `Model(vocab_size, cfg)`, whose `forward(batch)` returns logits. Build the network with
     `hale_llm.nn.backbones.build_backbone(vocab_size, cfg.model, ffn_type=...)` or compose the
     blocks in `hale_llm.nn` yourself.
   - `loss.py`: `@register_loss("<name>")`, `(model, batch) -> scalar`. Use `model_token_nll` so
     `train.loss_type` keeps working.
   - `sample.py`: `@register_sampler("<name>")`, returns the generated ids; stop at EOS.
   - the collate function, registered with `register_collate("<name>")`.
2. Import the package in `models/__init__.py`.
3. Metrics that apply to the variant: pass its name in `variants=` when registering them.
4. Add `configs/<name>/nano.yaml` (`inherits:` a base) and a unit test that runs one training step
   and one sample (see `tests/unit/test_transformer_variants.py`).

The reference is `models/models_autoregressive/`, which registers two variants from one class.
