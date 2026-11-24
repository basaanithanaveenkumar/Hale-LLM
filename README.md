<div align="center">

# HALE-LLM

Compare **autoregressive**, **masked diffusion**, **block diffusion**, and **flow matching** on one transformer backbone.

[![GitHub](https://img.shields.io/badge/GitHub-HaloDiffusionLLM-181717?logo=github&logoColor=white)](https://github.com/basaanithanaveenkumar/HaloDiffusionLLM)
[![Project Page](https://img.shields.io/badge/🌐_Project-Page-4A90D9)](https://basaanithanaveenkumar.github.io/HaloDiffusionLLM/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)

</div>

<p align="center">
  <img src="assets/block_diffusion_step4900.gif" alt="Block diffusion decoding on WikiText — tokens unmask block by block" width="100%">
</p>

<p align="center">
  <em>Block diffusion at step 4900: prompt in steelblue, committed tokens in orange, the active block in black on a light overlay.</em>
</p>

## Documentation

All explanations live in [`docs/`](docs/README.md):

- [Overview](docs/overview.md): how it works, design, variants
- [Getting started](docs/getting_started.md): install, train, sample, experiments, configuration, tests
- [Notebooks and resources](docs/notebooks.md): paper, project page, step-by-step notebooks
- [Configuration reference](docs/config.md) · [Architecture](docs/architecture.md) · [Deployment](docs/deploy.md)
