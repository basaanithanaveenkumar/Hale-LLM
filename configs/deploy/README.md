# Deployment configs

| path | what |
|------|------|
| `_shared/<variant>_adapter.yaml` | production adapter per variant (the `small` run configs; set `checkpoint`) |
| `_shared/tiny/<variant>_adapter.yaml` | CI-sized adapters (1 layer, vocab 128) |
| `<variant>/<target>.yaml` | variant × target, inheriting `pkg://hale_core.deploy/configs/<vendor>/<target>.yaml` |
| `ci/*.yaml` | tiny end-to-end pipelines run by the integration tests (ONNX Runtime, MLX fp32) |

Run: `hale-llm-deploy run configs/deploy/diffusion/tensorrt_fp16.yaml --cfg-options adapter.checkpoint=data/checkpoints/last.pt`.
See `docs/deploy.md`.
