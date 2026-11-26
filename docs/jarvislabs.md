# Training on Jarvislabs (India GPUs)

`hale-llm-cloud` rents a [Jarvislabs](https://jarvislabs.ai) GPU, ships this repo to it, starts
`hale-llm-train` in the background, and lets you follow the log and download the results.

```mermaid
flowchart LR
  L["hale-llm-cloud launch"] --> C["create instance<br/>(SDK, region IN2)"]
  C --> U["upload project files<br/>(jl upload, .tgz)"]
  U --> I["uv sync on the instance"]
  I --> T["nohup hale-llm-train<br/>writes train.log"]
  T --> F["logs / status"]
  F --> D["fetch -> data/cloud_runs/ID"]
  D --> P["pause (keep disk)<br/>or destroy"]
```

## One-time setup

1. Create an account, add wallet credit, and create an API key at <https://jarvislabs.ai/settings/api-keys>.
2. Install the client and store the key (the key is saved on your machine, never in the repo):

   ```bash
   uv sync --extra cloud
   uv run jl setup            # or: export JL_API_KEY=...
   ```

## Run a training

```bash
# Check the config on your own machine first (3 steps, reports params, loss, memory).
uv run hale-llm-train --config configs/autoregressive_moe/nano.yaml --dry-run

# Rent a GPU, upload the code, start training. Flags after the config go to hale-llm-train.
uv run hale-llm-cloud launch --config configs/autoregressive_moe/nano.yaml --steps 2000
```

Add `--auto-pause` to `launch` and it stays connected: when training ends (or crashes) it downloads the
results and **pauses the instance**, so billing stops without you. If the download fails it still pauses
(the disk is kept). Your terminal must stay open and the laptop awake; if you disconnect, reattach with
`hale-llm-cloud finish <machine_id>` (the same wait, download and pause).

`launch` prints a `machine_id`. Use it for everything else:

```bash
uv run hale-llm-cloud status  <machine_id>          # Running / Paused, cost, is training alive
uv run hale-llm-cloud logs    <machine_id> --tail 60
uv run hale-llm-cloud finish  <machine_id>          # wait for the end, download, pause
uv run hale-llm-cloud fetch   <machine_id> --pause  # download results, then stop compute billing
uv run hale-llm-cloud destroy <machine_id> --yes    # delete the instance and its disk
```

Results land in `data/cloud_runs/<machine_id>/` (checkpoints, `tb/` TensorBoard logs, tokenizer copy),
so nothing under `data/experiments/` is overwritten. View curves with
`uv run tensorboard --logdir data/cloud_runs/<machine_id>`.

Reuse a paused or running instance (skips renting; re-uploads the code):
`hale-llm-cloud launch --config ... --machine-id <machine_id>`.

## Settings

`configs/cloud/jarvislabs.yaml` (pass another file with `--cloud`):

| Key | Meaning |
|---|---|
| `gpu_type` | A30, L4, A100, A100-80GB, H100, H200, RTX-PRO6000 |
| `num_gpus` | GPUs on the instance (the lean trainer uses one) |
| `region` | `IN2` (L4, A30, A100, A100-80GB, H100, H200) or `IN1` (RTX-PRO6000); `null` lets Jarvislabs pick |
| `spot` | `true` rents a cheaper spot GPU (container templates only); default `false` is on-demand |
| `storage_gb` | Disk size |
| `remote_dir` | Where the code is unpacked on the instance |

## Precision study: fp32 vs bf16 vs fp16

`configs/precision_study/` trains the same ~110M-parameter dense model three times
(`fp32.yaml`, `bf16.yaml`, `fp16.yaml`). They inherit `base.yaml` and differ only in
`train.lean.precision` and the run name, with the same seed, so the curves are comparable.
`base.yaml` uses micro-batch 8 with 2 accumulation steps (16 x 1024 = 16,384 tokens per step). A first
version with micro-batch 16 ran out of memory in the fp32 run on a 24 GB A30. Run the fp32 dry run on the
GPU first (it prints peak CUDA memory), then raise `batch_size` and lower `accum_steps` together, keeping
their product at 16, until the peak is about 85 to 90%. All three runs must use the same values.

`hale-llm-cloud launch` always adds `--device cuda`: the shared configs say `device: mps` for the Mac, and
on a CUDA machine that would silently fall back to the CPU.

## Spot instances

Set `spot: true` in the cloud YAML (or copy it to `configs/cloud/jarvislabs_spot.yaml` and pass
`--cloud`). Check spot prices and free spot GPUs with `uv run jl gpus`.

**Spot interruptions are survived.** Jarvislabs can pause a spot instance when demand rises. With `spot: true`:

* `launch` adds `--resume` to the training command (spot only; on-demand runs keep the config's setting),
  and the trainer writes `resume.pt` every `train.lean.save_every` steps (default 2000).
* With `--auto-pause`, a supervisor watches the run. If the instance is paused underneath it, it **resumes the
  instance as spot** (waiting up to `resume_wait_minutes` for a free GPU), re-uploads the code, and restarts
  training, which continues from `resume.pt`. An interruption costs at most `save_every` steps. The instance
  id can change on resume; the supervisor follows it and prints the final one.
* It gives up after `max_restarts` interruptions, and it will not restart an instance that is not spot.
* After an interruption you can also do it by hand: `hale-llm-cloud launch --config ... --machine-id <id>`
  resumes a paused instance and continues from its checkpoint.
* Recovery needs the `--auto-pause` process to keep running on your machine. `hale-llm-cloud finish <id>` only waits
  and downloads; it cannot restart training because it does not know the arguments.

Settings in the cloud YAML: `max_restarts` (default 10) and `resume_wait_minutes` (default 180).

## What gets uploaded

Every file git does not ignore: tracked files with your working-tree edits, plus new files you have not
committed yet. `.gitignore`d files (virtualenv, checkpoints, `data/experiments/`) and `ext_pkgs/`, `assets/`,
`technical_report/`, `project-page/` and `data/cloud_runs/` are skipped. The shared tokenizer under `data/tokenizers/` is tracked, so the
instance uses the same token ids as your machine.

## Costs and cautions

* A running instance bills per hour. **`pause` stops compute billing but keeps a small storage charge;
  `destroy` deletes everything.** Pausing needs your Jarvislabs API key, which stays on your machine, so
  the instance cannot pause itself. Use `launch --auto-pause` or `finish`, or run `fetch --pause` yourself.
  If your laptop sleeps before training ends, the instance keeps billing.
* Check prices with `uv run jl gpus`.
* `uv sync` on the instance installs the locked PyTorch build, which is a large download on first launch.
* The lean trainer is single-GPU. TorchTitan multi-GPU runs are not wired into this command.
* If training fails, `launch --auto-pause` / `finish` print the last log lines, skip the download, and still
  pause the instance. Fix the problem, then relaunch with `--machine-id` to reuse the same instance.
* The log is `<remote_dir>/train.log` on the instance; `logs` reads it. The training exit code is written to
  `train.exit` next to it, which is how the tool knows training has ended.

## Not verified

The code is unit-tested against a fake SDK and a fake `jl`. It has **not** been run against a real
Jarvislabs account here, so the first `launch` is the real test. Watch the first `logs` output.

## Code map

| Piece | File |
|---|---|
| Instance settings | `cloud/jarvislabs/settings.py` |
| Create / pause / destroy (SDK) | `cloud/jarvislabs/instance_manager.py` |
| `jl` exec / upload / download | `cloud/jarvislabs/jl_cli.py` |
| Tracked-files tarball | `cloud/jarvislabs/project_archive.py` |
| Deploy, start, logs, fetch | `cloud/jarvislabs/remote_training_job.py` |
| Command line | `cli/cloud.py` |
