"""Backend that launches TorchTitan on NVIDIA GPUs."""

from __future__ import annotations

import importlib.util
import os
import subprocess
from pathlib import Path

import torch
from loguru import logger

from hale_llm.config.schema import RunConfig
from hale_llm.training.backends.registry import register_trainer_backend
from hale_llm.training.backends.titan_command_builder import TorchTitanCommandBuilder
from hale_llm.training.backends.titan_errors import TorchTitanUnavailableError
from hale_llm.training.backends.trainer_backend import TrainerBackend


@register_trainer_backend("torchtitan")
class TorchTitanBackend(TrainerBackend):
    """Runs a TorchTitan recipe with ``torchrun``.

    Steps:

    1. check that CUDA GPUs exist (Mac/MPS/CPU raise :class:`TorchTitanUnavailableError`),
    2. check that ``torchtitan`` is importable and that the recipe directory exists,
    3. build the ``torchrun`` command from ``train.titan``,
    4. run it from the TorchTitan checkout (its recipes import as ``torchtitan_recipes``)
       and raise if it exits non-zero.

    TorchTitan builds its own model, data and parallelism from the recipe, so the
    ``model`` / ``data`` sections of this YAML are not used by this backend.
    """

    def validate(self, cfg: RunConfig) -> None:
        if not torch.cuda.is_available():
            raise TorchTitanUnavailableError(
                "train.trainer=torchtitan needs NVIDIA GPUs (torch.cuda.is_available() is False). "
                "On a Mac or CPU-only machine use train.trainer=lean."
            )
        if importlib.util.find_spec("torchtitan") is None:
            raise TorchTitanUnavailableError(
                "torchtitan is not installed. Run: uv pip install -e ext_pkgs/torchtitan "
                "-r ext_pkgs/torchtitan/requirements.txt"
            )
        repo = Path(cfg.train.titan.repo_dir)
        if not (repo / "torchtitan_recipes").is_dir():
            raise TorchTitanUnavailableError(
                f"{repo}/torchtitan_recipes not found. Run: git submodule update --init ext_pkgs/torchtitan "
                "or set train.titan.repo_dir"
            )

    def run(self, cfg: RunConfig, tokenizer=None) -> int:
        self.validate(cfg)
        settings = cfg.train.titan
        gpus = settings.num_gpus or torch.cuda.device_count()
        if gpus > torch.cuda.device_count():
            raise TorchTitanUnavailableError(
                f"train.titan.num_gpus={gpus} but only {torch.cuda.device_count()} GPU(s) visible"
            )
        command = TorchTitanCommandBuilder(settings, gpus).build()
        repo = Path(settings.repo_dir).resolve()
        logger.info("training backend: torchtitan | {} GPU(s) | recipe {}:{}", gpus, settings.module, settings.config)
        logger.info("running in {}: {}", repo, " ".join(command))

        # Same allocator setting as TorchTitan's run_train.sh; the caller may override it.
        env = {**os.environ, "PYTORCH_ALLOC_CONF": os.environ.get("PYTORCH_ALLOC_CONF", "expandable_segments:True")}
        result = subprocess.run(command, cwd=repo, env=env, check=False)
        if result.returncode != 0:
            logger.error("torchtitan exited with code {}", result.returncode)
            raise RuntimeError(f"torchtitan failed with exit code {result.returncode}")
        return result.returncode
