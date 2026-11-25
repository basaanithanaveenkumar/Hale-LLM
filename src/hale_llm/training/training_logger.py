"""Writes the training run to TensorBoard (and stays silent when TensorBoard is off).

Everything the lean trainer wants to see over time goes through this class, so the
training loop itself only says *what* happened, not how it is stored.

Tags, grouped the way TensorBoard shows them:

    train/    loss (what is optimised), ce_loss (next-token loss alone), perplexity,
              bits_per_token, moe_balance_loss, grad_norm, grad_clipped, lr, lr_scale,
              tokens_seen, tokens_per_sec, step_seconds, epoch
    val/      loss, perplexity, bits_per_token
    moe/      layerN/expert_share_max, layerN/expert_share_min, expert_share_max, expert_share_min
    system/   gpu_memory_gb (CUDA only)
    model/    parameters, weight_norm
    context/  window (the context length of the current stage)
"""

from __future__ import annotations

import math
from pathlib import Path

import torch
import yaml
from loguru import logger
from torch import nn

from hale_llm.config.schema import RunConfig
from hale_llm.utils.tensorboard import log_scalars, make_writer


class TrainingLogger:
    """A thin wrapper around a TensorBoard ``SummaryWriter`` (or nothing, if disabled)."""

    def __init__(self, cfg: RunConfig) -> None:
        self._writer = make_writer(cfg.logging.tensorboard_dir, enabled=cfg.logging.tensorboard)

    @property
    def enabled(self) -> bool:
        return self._writer is not None

    def __enter__(self) -> TrainingLogger:
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    def log_run_start(self, cfg: RunConfig, model: nn.Module, total_steps: int, windows: int) -> None:
        """Record what is being trained: the full config as text and the size of the model."""
        if self._writer is None:
            return
        params = sum(p.numel() for p in model.parameters())
        self._writer.add_text("run/config", "```yaml\n" + yaml.safe_dump(cfg.model_dump(), sort_keys=False) + "\n```", 0)
        log_scalars(self._writer, 0, {"model/parameters": params, "run/total_steps": total_steps, "run/train_windows": windows})

    def log_step(self, step: int, scalars: dict[str, float]) -> None:
        """Log scalars for one optimizer step (``step`` counts from 1)."""
        log_scalars(self._writer, step, scalars)

    def log_validation(self, step: int, loss: float) -> None:
        """Validation loss plus the numbers people actually compare: perplexity and bits per token."""
        log_scalars(self._writer, step, {
            "val/loss": loss,
            "val/perplexity": math.exp(min(loss, 20.0)),
            "val/bits_per_token": loss / math.log(2),
        })
        self.flush()

    def log_slow_stats(self, step: int, model: nn.Module, device: str) -> None:
        """Statistics that cost a pass over the weights; called every ``log_every`` steps."""
        if self._writer is None:
            return
        with torch.no_grad():
            weight_norm = torch.linalg.vector_norm(torch.stack([p.norm() for p in model.parameters()]))
        scalars = {"model/weight_norm": float(weight_norm)}
        if device == "cuda":
            scalars["system/gpu_memory_gb"] = torch.cuda.max_memory_allocated() / 1e9
        scalars.update(self._expert_load(model))
        log_scalars(self._writer, step, scalars)

    @staticmethod
    def _expert_load(model: nn.Module) -> dict[str, float]:
        """Share of the routing decisions each expert got, per layer.

        With ``E`` experts an even split is ``1 / E``. A ``max`` far above that, or a ``min``
        near 0, means the router is collapsing onto a few experts.
        """
        scalars: dict[str, float] = {}
        maxima, minima = [], []
        backbone = getattr(model, "backbone", None)
        layers = backbone.stack.layers if backbone is not None else []
        for index, layer in enumerate(layers):
            share = getattr(layer.ffn, "expert_share", None)
            if share is None:
                continue
            scalars[f"moe/layer{index}/expert_share_max"] = float(share.max())
            scalars[f"moe/layer{index}/expert_share_min"] = float(share.min())
            maxima.append(float(share.max()))
            minima.append(float(share.min()))
        if maxima:
            scalars["moe/expert_share_max"] = max(maxima)
            scalars["moe/expert_share_min"] = min(minima)
        return scalars

    def flush(self) -> None:
        if self._writer is not None:
            self._writer.flush()

    def close(self) -> None:
        if self._writer is not None:
            self._writer.close()
            self._writer = None
            logger.info("tensorboard logs written")
