"""Mixed-precision training for the lean trainer: fp32, bf16 or fp16.

Weights and optimizer state always stay in fp32, so checkpoints are identical in every mode.
Only the forward pass runs in lower precision (``torch.autocast``).

    fp32  no autocast. Slowest, the reference.
    bf16  autocast to bfloat16. Same exponent range as fp32, so no loss scaling is needed.
    fp16  autocast to float16. Its range is small, so small gradients would round to zero;
          the loss is multiplied by a scale first (``GradScaler``) and the gradients are
          divided by it again before clipping. A step whose gradients overflow is skipped
          and the scale is halved.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterable

import torch
from loguru import logger

_DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16}


class MixedPrecision:
    """Autocast plus (for fp16) loss scaling, behind one small interface."""

    def __init__(self, device_type: str, dtype: torch.dtype | None) -> None:
        """``dtype=None`` means plain fp32. ``device_type`` is ``cuda``, ``cpu`` or ``mps``."""
        self.device_type = device_type
        self.dtype = dtype
        # GradScaler(enabled=False) passes everything through, so the training loop needs no
        # "if fp16" branches: backward and step always go through it.
        self.scaler = torch.amp.GradScaler(device_type, enabled=dtype == torch.float16)
        self.step_skipped = False  # True when the last update was skipped for overflowing gradients

    @classmethod
    def from_setting(cls, precision: str, device: str) -> "MixedPrecision":
        """Turn the YAML value (``auto | fp32 | bf16 | fp16``) into a configured instance."""
        # "auto" keeps the original behaviour: bf16 on a CUDA GPU that supports it, fp32 elsewhere.
        if precision == "auto":
            precision = "bf16" if device == "cuda" and torch.cuda.is_bf16_supported() else "fp32"
        if precision == "bf16" and device == "cuda" and not torch.cuda.is_bf16_supported():
            raise ValueError("this GPU does not support bf16; use train.lean.precision: fp16")
        mixed = cls(device, _DTYPES.get(precision))
        logger.info("precision={} on {} (loss scaling: {})", precision, device, mixed.scaler.is_enabled())
        return mixed

    @property
    def name(self) -> str:
        """``fp32``, ``bf16`` or ``fp16``."""
        return "fp32" if self.dtype is None else {v: k for k, v in _DTYPES.items()}[self.dtype]

    def autocast(self):
        """Context manager for the forward pass (and for validation)."""
        if self.dtype is None:
            return contextlib.nullcontext()
        return torch.autocast(device_type=self.device_type, dtype=self.dtype)

    def backward(self, loss: torch.Tensor) -> None:
        """Backpropagate; with fp16 the loss is scaled up first so small gradients survive."""
        self.scaler.scale(loss).backward()

    def clip_and_step(self, optimizer, parameters: Iterable[torch.Tensor], max_norm: float) -> float:
        """Unscale gradients, clip their global norm, update the weights. Returns the true grad norm.

        With fp16, an overflow (inf/nan gradients) makes the scaler skip the update and lower
        the scale; ``step_skipped`` says so and the returned norm is then inf/nan.
        """
        # Step 1: undo the loss scale so the norm and the clip threshold are in real units.
        self.scaler.unscale_(optimizer)
        # Step 2: clip the global gradient norm.
        grad_norm = float(torch.nn.utils.clip_grad_norm_(parameters, max_norm))
        # Step 3: update the weights (skipped by the scaler if gradients overflowed), then adapt the scale.
        scale_before = self.scaler.get_scale()
        self.scaler.step(optimizer)
        self.scaler.update()
        self.step_skipped = self.scaler.get_scale() < scale_before
        return grad_norm

    @property
    def loss_scale(self) -> float:
        """Current fp16 loss scale (1.0 when scaling is off)."""
        return self.scaler.get_scale()
