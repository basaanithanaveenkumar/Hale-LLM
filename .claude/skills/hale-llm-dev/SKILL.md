---
name: hale-llm-dev
description: Set up, navigate and test HALE-LLM (`ha_llm`), which compares autoregressive, masked diffusion, block diffusion, flow matching and multi-token prediction on one transformer backbone. Use when starting work in this repo, touching core/ components or a models_* paradigm, or when tests fail to import.
---

# HALE-LLM development

Default branch: `masked_diffusion_llm`. Package: `ha_llm` (Python ≥ 3.12).

```bash
uv sync --extra dev
uv run pytest tests/unit -q
```

## Test status (checked 2026-09)

`tests/unit`: 53 pass. Three tests still import the old package name
`halodiffusionllm` and fail at import (`test_import.py`, `test_block_mask.py`,
`test_embeddings.py`). Port them to `ha_llm` (the mask is now
`ha_llm.core.components.attention.masks.build_attn_mask("block_causal", ...)`; the time
embedding is in `ha_llm.core.components.embeddings`) rather than deleting them. Until then,
run:

```bash
uv run pytest tests/unit -q --ignore=tests/unit/test_block_mask.py --ignore=tests/unit/test_embeddings.py
```

## Layout

| Path | What |
|---|---|
| `core/components/` | attention (`mha`, `gqa`, `mqa`, RoPE, masks, window), FFN (`mlp`, `geglu`, `moe`), `RMSNorm`, embeddings, `TransformerLayer`, `DiTBlock` |
| `core/transformers/` | `GPTStack`, `LGTStack`, `DiTStack` over hidden states `(B, S, D)` |
| `core/backbones/` | GPT / LGT / DiT token LMs (ids → logits), `build_backbone` |
| `core/registry.py` | `register_variant`, `register_loss`, `register_sampler`, `register_metric` |
| `models/models_<paradigm>/` | `autoregressive`, `diffusion`, `block_diffusion`, `flow_matching`, `mtp` (stub): model, loss, sample, metrics |
| `losses/` | token losses `ce`, `focal`, `label_smoothing`, `kl` |
| `metrics/llm/` | loss, perplexity, bits/token, token accuracy, top-k |
| `dataloader/` | GPT-2 tokenizer (+ `[MASK]`), sliding windows (`stride_words`), HF / overfit sources, per-variant collate registry |
| `training/`, `evaluation/`, `inference/`, `visualization/` | trainer, evaluator, `load_session`/`generate`, unmasking GIFs |
| `cli/` | `ha-llm-train`, `ha-llm-sample`, `ha-llm-eval`, `ha-llm-viz` |
| `src/masked_diffusion/` | early standalone prototype (not used by `ha_llm`) |

`core/` mirrors HaleBlocks' `hale_core.nn` but is vendored. It has no HaleBlocks
dependency.

**Rule:** shared code never imports a `models_*` package by name. Paradigms plug in via
registries, plus one import in `models/__init__.py`.

## Sizes (measured, `configs/base.yaml` model, GPT-2 vocab 50,258)

AR GPT backbone: 47.7M. Diffusion / flow-matching LGT backbone (GQA n_kv = 2, GeGLU, time
conditioning): 63.2M.
