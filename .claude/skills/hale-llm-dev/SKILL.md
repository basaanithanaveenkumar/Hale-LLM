---
name: hale-llm-dev
description: Set up, navigate and test HALE-LLM (`hale_llm`), a small decoder-only transformer codebase with a dense, a mixture-of-experts and a multi-token-prediction variant, a tokenizer trained on the data, and a lean trainer. Use when starting work in this repo, touching nn/ blocks, a models_* variant, the tokenizer or the trainer, or when tests fail to import.
---

# HALE-LLM development

**Working rules:** follow the behavioral guidelines in `/CLAUDE.md` (think before coding, simplicity first, surgical changes, goal-driven execution) on every task.

Package: `hale_llm` (Python >= 3.12). There is no external block library: the registries, the
checkpoint store and the optimizer base live in this repo.

```bash
uv sync --extra dev
uv run pytest claude_tests -q
```

## Layout

| Path | What |
|---|---|
| `nn/attention`, `nn/blocks`, `nn/moe`, `nn/backbones` | `HeadAttention`, `MultiHeadAttention`, `TransformerBlock`, `build_blocks`, `FeedForward`, routers, `Expert`, `DeepseekMoE`, `auxiliary_loss` (the MoE balancing loss of any module), `TransformerBackbone` (embedding, `layers`, `norm`, head) |
| `nn/positional`, `nn/norm` | RoPE (+ linear / YaRN scaling) and learned positions; LayerNorm, RMSNorm; each behind a registry |
| `models/models_autoregressive/` | variants `autoregressive` and `autoregressive_moe`: model, loss, sampler, collate |
| `nn/mtp`, `models/models_mtp/` | DeepSeek-V3 style multi-token prediction: `MTPModule`, `MultiTokenPredictor`, and the opt-in offset-conditioned `OffsetConditionedPredictor` with its `StratifiedOffsetAssigner`; variants `autoregressive_mtp` and `autoregressive_moe_mtp` (loss, lambda schedule, speculative-decoding sampler) |
| `registry/` | `NamedRegistry`, `VariantRegistry`, `register_variant/loss/sampler/metric` |
| `runtime/` | `CheckpointStore`, `count_parameters`, `log_model_summary`, `move_batch_to_device` |
| `optim/` | `adam`, `adamw` (LLM recipe), `sgd`, `muon`, `dion` |
| `losses/`, `metrics/` | token losses (`ce`, `focal`, `label_smoothing`, `kl`); loss, perplexity, bits/token, accuracy, generation metrics |
| `tokenizer/`, `dataloader/` | trained SuperBPE/BPE tokenizer and vocab policy; non-overlapping context windows with EOS between documents |
| `training/`, `inference/`, `chat/`, `cli/` | lean trainer + TorchTitan launcher, `load_session`, chat loop, CLIs |

**Rule:** shared code never imports a `models_*` package by name. Variants plug in through the
registries plus one import in `models/__init__.py`.

Docs: `docs/architecture.md` (diagrams), `docs/mtp.md`, `docs/config.md`, `docs/tokenizer.md`, `docs/context_length.md`.
