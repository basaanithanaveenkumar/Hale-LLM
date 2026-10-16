# ext_pkgs - vendored third-party sources

Shallow git submodules (`shallow = true` in `.gitmodules`, LFS smudging skipped). They are
**reference sources** - samples, converters, op-support tables, version pinning - not build
dependencies: the deploy stages use the pip / SDK installs (`pip install -e ".[deploy-*]"`).

```bash
git submodule update --init --depth 1 ext_pkgs/<name>   # never --recursive (executorch pulls GBs)
```

| name | upstream | vendor | why it is vendored | used by (hale_core.deploy stage) |
|---|---|---|---|---|
| `grokfast` | ironjr/grokfast | - | training optimizer (pre-existing) | training |
| `TensorRT` | NVIDIA/TensorRT | NVIDIA | ONNX parser, plugins, samples, `trtexec`, TRT 10→11 migration | `TensorRTEngineStage` |
| `TensorRT-LLM` | NVIDIA/TensorRT-LLM | NVIDIA | PyTorch-backend LLM serving, `trtllm-serve` options | `TensorRTLLMBundleStage` |
| `TensorRT-Edge-LLM` | NVIDIA/TensorRT-Edge-LLM | NVIDIA | on-device LLM/VLM runtime for Jetson Thor / Orin | Jetson LLM path (docs) |
| `Model-Optimizer` | NVIDIA/Model-Optimizer | NVIDIA | AutoCast, ONNX PTQ (INT8/FP8/INT4), HF checkpoint export | `ModelOptAutoCastStage`, `ModelOptQuantizeStage` |
| `Torch-TensorRT` | pytorch/TensorRT | NVIDIA / PyTorch | dynamo IR partial TensorRT lowering | `TorchTensorRTStage` |
| `mlx` | ml-explore/mlx | Apple | ops, `export_function` / `mx.exporter`, fast kernels | `MLXConvertStage`, `MLXFunctionSession` |
| `mlx-lm` | ml-explore/mlx-lm | Apple | HF → MLX LLM conversion and serving | `MLXHuggingFaceConvertStage` |
| `AMDMIGraphX` | ROCm/AMDMIGraphX | AMD | ONNX op support, `migraphx-driver`, quantization | `MIGraphXCompileStage`, `MIGraphXSession` |
| `Quark` | amd/Quark | AMD | ONNX PTQ presets (A16W8, BF16) for Ryzen AI | `QuarkQuantizeStage` |
| `RyzenAI-SW` | amd/RyzenAI-SW | AMD | Vitis AI EP examples, NPU deployment recipes | `OnnxRuntimePackageStage` (Vitis AI EP) |
| `ai-hub-models` | qualcomm/ai-hub-models | Qualcomm | AI Hub compile/profile recipes, device list, Genie apps | `QAIHubCompileStage` |
| `executorch` | pytorch/executorch | PyTorch / Qualcomm | QNN backend, `QnnQuantizer`, Android runtime | `ExecuTorchQNNStage` |
