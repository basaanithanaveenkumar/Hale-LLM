---
name: hale-llm-train
description: Train, sample, evaluate and chat with HALE-LLM models (dense or mixture-of-experts transformer) on TinyStories with the hale-llm CLIs, and manage experiment folders. Use when asked to run or compare the variants, tune configs, or try a trained model.
---

# Running HALE-LLM

```bash
uv run hale-llm-train  --config configs/autoregressive/small.yaml --steps 2000 --no-resume
uv run hale-llm-train  --config configs/autoregressive_moe/nano.yaml --steps 2000 --no-resume
uv run hale-llm-sample --config configs/autoregressive/small.yaml
uv run hale-llm-eval   --config configs/autoregressive/small.yaml
uv run hale-llm-chat   --config configs/autoregressive/small.yaml --max-new-tokens 40
tensorboard --logdir data/experiments
```

## Configs

`configs/base.yaml` holds the shared model, data, tokenizer, logging and experiment settings;
`configs/autoregressive/{nano,small}.yaml` set the dense model, `configs/autoregressive_moe/nano.yaml`
the mixture of experts. `configs/long_context/` and `configs/ablations/` hold the 4k -> 32k recipe and
the BPE vs SuperBPE ablation. Every run writes `data/experiments/<variant>/<variant>_<stamp>/` with
`config.yaml`, `checkpoints/last.pt`, `tokenizer/`, `logs/` and `tb/`; `latest` points at the newest.

Each run trains its own tokenizer on the training split (see `docs/tokenizer.md`); sample, eval and
chat load the one saved beside the checkpoint.

Full key reference: `docs/config.md`.
