---
name: hale-llm-train
description: Train, sample, evaluate and visualise HALE-LLM paradigms (autoregressive, masked diffusion, block diffusion, flow matching) on WikiText-2 with the ha-llm CLIs, manage experiment folders and resume. Use when asked to run or compare paradigms, tune configs, or produce unmasking GIFs.
---

# Running HALE-LLM

```bash
uv run ha-llm-train  --config configs/block_diffusion/small.yaml --no-resume --viz
uv run ha-llm-sample --config configs/block_diffusion/small.yaml
uv run ha-llm-eval   --config configs/block_diffusion/small.yaml
uv run ha-llm-viz    --config configs/block_diffusion/small.yaml
tensorboard --logdir data/experiments
```

## Configs

`configs/base.yaml` (d_model 384, 16 layers, 8 heads, d_ff 1536, max_length 128, GPT-2
tokenizer, WikiText-2 raw with a 10-word sliding stride, `device: mps`), plus one file per
paradigm:

| Config | Backbone | Mask | Notes |
|---|---|---|---|
| `autoregressive/small.yaml` | `transformer` | causal | shifted labels in the collate |
| `diffusion/small.yaml` | `lgt`, GQA n_kv = 2, GeGLU, time-conditioned | bidirectional | batch 24 |
| `block_diffusion/small.yaml` | `lgt` | block_causal, `block_size: 16` | `max_length % block_size == 0` |
| `flow_matching/small.yaml` | `lgt` | bidirectional | |
| `mtp/small.yaml` | — | causal | stub: only the main head is trained |

Set `device: cuda` (or `cpu`) for non-Mac machines. Unknown keys are errors. See
`docs/config.md` for every key.

## Objectives (what the code does)

| Paradigm | Corruption | Loss |
|---|---|---|
| AR | none | CE(next token) |
| Masked diffusion | mask each token with prob `t ~ U(ε, 1)` (α_t = 1 − t) | CE on masked positions × 1/t |
| Block diffusion | per-block `t`; clean ‖ noisy input with block-causal mask | CE on masked noisy positions × 1/t |
| Flow matching | mask with prob `1 − t`, `t ~ U(0, 1 − ε)` | CE on masked positions × 1/(1 − t), equivalent to masked diffusion under a time flip |

Samplers: diffusion reveals each masked token with probability `(t − s)/t` per step
(`sample.sampling_steps`). Block diffusion goes block by block and runs
`sampling_steps` steps inside each block. AR uses standard temperature sampling.

## Experiments

Runs land in `data/experiments/<variant>/<variant>_<timestamp>/` with `config.yaml`,
the source YAMLs, `checkpoints/last.pt`, `logs/`, `tb/` and `viz/`. `latest` is a symlink.
Resume with `--resume` (latest) or `--experiment NAME`.

## Comparing paradigms fairly

Keep `model` size, data (`train_size`, `stride_words`), steps and `eval.metrics` equal.
Perplexity for diffusion models is an ELBO bound, not an exact likelihood, so report
it as such. `masked_accuracy` only applies to the masking paradigms.
