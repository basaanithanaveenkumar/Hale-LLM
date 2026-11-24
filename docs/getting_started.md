# Getting started

## Quick start

```bash
uv sync
uv sync --extra dev
uv run pytest tests/unit -q
```

Train block diffusion on full WikiText-2 (MPS / CUDA via `device` in YAML):

```bash
uv run hale-llm-train --config configs/block_diffusion/small.yaml --no-resume --viz
```

Sample, eval, and write a GIF from `latest` or a named run:

```bash
uv run hale-llm-sample --config configs/block_diffusion/small.yaml
uv run hale-llm-eval --config configs/block_diffusion/small.yaml
uv run hale-llm-viz --config configs/block_diffusion/small.yaml
tensorboard --logdir data/experiments
```

## Experiments

Each run is namespaced:

```
data/experiments/<variant>/<variant>_<YYYYMMDD_HHMMSS>/
  config.yaml          # resolved settings
  configs/             # source YAML chain
  checkpoints/last.pt
  logs/train.log
  tb/                  # TensorBoard
  viz/                 # generation GIFs
```

`latest` is a symlink to the newest folder. Resume with `--resume` (uses `latest`) or `--experiment NAME`.

Training GIFs write every `viz.every_n_steps` (default 50) using a short sampler. Prompts are taken from WikiText validation (`viz.prompt_source: dataset`). Override with `--prompt`.

## Configuration

YAML files under `configs/` are the source of truth. [`configs/base.yaml`](../configs/base.yaml) holds shared size, data, and logging. Each variant file sets `inherits: ../base.yaml` and overrides only what that paradigm needs (`variant`, attention mask, `arch`, `block_size`).

Unknown keys are rejected. After a run, the merged file is saved as `data/experiments/<variant>/<run>/config.yaml`.

**High-level map**

| Section | Role |
|---|---|
| `variant` | Which model/loss/sampler (`autoregressive`, `diffusion`, `block_diffusion`, `flow_matching`, `block_flow_matching`, `looped`, `mtp`) |
| `model` | Width/depth, `arch` (`transformer` \| `lgt` \| `dit` \| `looped`, with `model.loop`), mask (`causal` \| `bidirectional` \| `block_causal`), FFN, time cond |
| `train` | `epochs` or `steps`, `batch_size`, `lr`, `loss_type`, resume |
| `data` | Hugging Face dataset (`Salesforce/wikitext` + `subset`) or `overfit_text`; `train_size: null` means the **full** split |
| `sample` / `eval` / `viz` | Generation length, metrics, GIF cadence |
| `device` | `mps`, `cuda`, or `cpu` |

**Typical edits**

```yaml
# WikiText-2, full split (null = no row cap)
data:
  dataset: Salesforce/wikitext
  subset: wikitext-2-raw-v1
  train_size: null

# Swap LGT for DiT on a denoiser
model:
  arch: dit

# Faster laptop run
train:
  batch_size: 8
  epochs: 2
data:
  train_size: 512
```

**Scaling out: Ray Train + WebDataset** (both off by default, `uv sync --extra distributed`)

```yaml
train:
  ray: {enabled: true, num_workers: 4, use_gpu: true}   # run on Ray workers with DDP, fault tolerant
data:
  webdataset: {enabled: true, root: data/webdataset}    # tokenise once -> tar shards -> stream
```

Full example: [`configs/examples/ray_webdataset.yaml`](../configs/examples/ray_webdataset.yaml) ·
details: [docs/distributed.md](distributed.md).

Every key, type, default, and recipe: **[docs/config.md](config.md)**.

## Adding a paradigm

1. New folder `src/hale_llm/models/models_<name>/` with model, loss, sampler (and optional metric).
2. One import in `src/hale_llm/models/__init__.py`.
3. One YAML under `configs/<name>/`.

Public facades stay the same: `train()`, `evaluate()`, `load_session()`.

## Deployment

Export any variant to TensorRT (data center, RTX, Jetson), MLX (Mac), AMD (MIGraphX, Ryzen AI
NPU) and Qualcomm (AI Hub, ExecuTorch QNN) with `hale-llm-deploy run configs/deploy/<variant>/<target>.yaml`.
Adapters, per-variant contracts, supported targets and research notes: [docs/deploy.md](deploy.md).

## Tests

```bash
uv run pytest tests/unit -q
uv run pytest tests/integration/test_overfit_sanity.py -k autoregressive -v
```
