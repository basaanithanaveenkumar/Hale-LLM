"""Pick the compute device for training / inference: CPU, NVIDIA CUDA, AMD ROCm or Apple MPS.

One small, dependency-free (``torch`` only) module that every Hale* repo carries an identical copy of.

* **CUDA and ROCm.** PyTorch's ROCm (AMD) wheels expose the GPU through the ``cuda`` namespace:
  ``torch.cuda.is_available()`` is ``True`` and ``torch.version.hip`` is set. So an AMD GPU *is* the device
  ``"cuda"`` to torch; :func:`detect_backend` just reports it as ``"rocm"`` so logs and tests can tell them apart,
  and ``"rocm"`` / ``"amd"`` / ``"hip"`` are accepted as request names.
* **Fallback, never a crash.** An unavailable request falls back (with a warning) to the best present device,
  because a failure at the first ``.to("cuda")`` is a worse error than a slower run. ``strict=True`` raises instead.
* **Distributed.** :func:`dist_backend` returns ``"nccl"`` for CUDA *and* ROCm (RCCL answers to the same name)
  and ``"gloo"`` for CPU / MPS.

Typical use::

    from <pkg>.device import get_device, add_device_argument
    device = torch.device(get_device(args.device))      # "auto" | "cpu" | "cuda" | "cuda:1" | "rocm" | "mps" | "gpu"
"""

from __future__ import annotations

import logging
import os
import random
from typing import Any

import torch

__all__ = [
    "DeviceError",
    "add_device_argument",
    "available_backends",
    "detect_backend",
    "device_summary",
    "dist_backend",
    "empty_cache",
    "get_device",
    "manual_seed_all",
    "resolve_device",
    "synchronize",
]

_LOG = logging.getLogger(__name__)

_CUDA_NAMES = {"cuda"}
_ROCM_NAMES = {"rocm", "amd", "hip"}
_GPU_NAMES = {"gpu", "accelerator"}
_AUTO_NAMES = {"", "auto", "best", "default"}


class DeviceError(RuntimeError):
    """A strictly requested device is unavailable, or the request is not a device name."""


# --------------------------------------------------------------------------------------------------
# detection
# --------------------------------------------------------------------------------------------------
def _cuda_usable() -> bool:
    return bool(torch.cuda.is_available())


def _is_hip_build() -> bool:
    return getattr(torch.version, "hip", None) is not None


def _mps_usable() -> bool:
    mps = getattr(torch.backends, "mps", None)
    return bool(mps is not None and mps.is_available())


def available_backends() -> list[str]:
    """Backends usable right now, best first: ``cuda`` *or* ``rocm``, then ``mps``, always ``cpu``."""
    out: list[str] = []
    if _cuda_usable():
        out.append("rocm" if _is_hip_build() else "cuda")
    if _mps_usable():
        out.append("mps")
    out.append("cpu")
    return out


def detect_backend() -> str:
    """The best backend present: ``"cuda"``, ``"rocm"``, ``"mps"`` or ``"cpu"``."""
    return available_backends()[0]


# --------------------------------------------------------------------------------------------------
# resolution
# --------------------------------------------------------------------------------------------------
def _fallback(reason: str, strict: bool) -> torch.device:
    if strict:
        raise DeviceError(reason)
    best = detect_backend()
    _LOG.warning("%s; falling back to %s", reason, best)
    return resolve_device(None)


def resolve_device(requested: str | torch.device | None = None, *, strict: bool = False) -> torch.device:
    """Turn a request into a usable :class:`torch.device`.

    Args:
        requested: ``None`` / ``"auto"`` (best present), ``"cpu"``, ``"cuda"`` / ``"cuda:N"``,
            ``"rocm"`` / ``"amd"`` / ``"hip"`` (AMD; the torch device is ``cuda``), ``"mps"``,
            ``"gpu"`` (best accelerator, else cpu), or a ``torch.device``.
        strict: raise :class:`DeviceError` instead of falling back when the request is unavailable.

    Raises:
        DeviceError: ``strict`` and unavailable, or an unknown device name.
    """
    if isinstance(requested, torch.device):
        requested = str(requested)
    name = (requested or "").strip().lower()
    if name in _AUTO_NAMES:
        backend = detect_backend()
        if backend in ("cuda", "rocm"):
            return torch.device("cuda")
        return torch.device(backend)
    if name in _GPU_NAMES:
        backend = detect_backend()
        if backend == "cpu":
            return _fallback("a GPU was requested but none is present", strict)
        return torch.device("cuda" if backend in ("cuda", "rocm") else backend)
    if name == "cpu":
        return torch.device("cpu")
    if name == "mps":
        if not _mps_usable():
            return _fallback("mps requested but not available", strict)
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")  # ops MPS lacks run on the CPU, not crash
        return torch.device("mps")

    base, _, index = name.partition(":")
    if base in _ROCM_NAMES:
        if not (_cuda_usable() and _is_hip_build()):
            return _fallback(f"{base} (AMD ROCm) requested but no ROCm GPU / ROCm build of torch", strict)
        base = "cuda"
    if base in _CUDA_NAMES:
        if not _cuda_usable():
            return _fallback("cuda requested but not available", strict)
        if index:
            if not index.isdigit():
                raise DeviceError(f"bad device index in {requested!r}")
            if int(index) >= torch.cuda.device_count():
                return _fallback(f"{requested} requested but only {torch.cuda.device_count()} GPU(s) present", strict)
            return torch.device("cuda", int(index))
        return torch.device("cuda")
    raise DeviceError(f"unknown device {requested!r}; use auto | cpu | cuda[:N] | rocm | mps | gpu")


def get_device(device: str | torch.device | None = None, *, strict: bool = False) -> str:
    """:func:`resolve_device` as a string (``"cuda"``, ``"cuda:1"``, ``"mps"`` or ``"cpu"``)."""
    return str(resolve_device(device, strict=strict))


def add_device_argument(parser: Any, default: str = "auto", *, flag: str = "--device") -> Any:
    """Add ``--device`` to an ``argparse`` parser; the value is resolved (and falls back) at parse time."""
    return parser.add_argument(
        flag,
        default=default,
        type=get_device,
        help="auto | cpu | cuda[:N] | rocm (AMD) | mps (Apple) | gpu; an unavailable choice falls back (default: %(default)s)",
    )


# --------------------------------------------------------------------------------------------------
# helpers that differ per backend
# --------------------------------------------------------------------------------------------------
def dist_backend(device: str | torch.device | None = None) -> str:
    """``torch.distributed`` backend for ``device``: ``nccl`` (CUDA and ROCm/RCCL) or ``gloo`` (CPU, MPS)."""
    dev = resolve_device(device)
    return "nccl" if dev.type == "cuda" else "gloo"


def synchronize(device: str | torch.device | None = None) -> None:
    """Block until queued work on ``device`` finishes (needed for honest timings); a no-op on CPU."""
    dev = resolve_device(device)
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)
    elif dev.type == "mps":
        torch.mps.synchronize()


def empty_cache(device: str | torch.device | None = None) -> None:
    """Release cached allocator memory on ``device``; a no-op on CPU."""
    dev = resolve_device(device)
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    elif dev.type == "mps":
        torch.mps.empty_cache()


def manual_seed_all(seed: int) -> None:
    """Seed python, torch (CPU) and every available accelerator."""
    random.seed(seed)
    torch.manual_seed(seed)
    if _cuda_usable():
        torch.cuda.manual_seed_all(seed)
    if _mps_usable():
        torch.mps.manual_seed(seed)


def device_summary() -> dict[str, Any]:
    """A loggable description of what is present (backend, device names, torch / HIP / CUDA versions)."""
    info: dict[str, Any] = {
        "torch": torch.__version__,
        "backends": available_backends(),
        "best": detect_backend(),
        "cuda_version": getattr(torch.version, "cuda", None),
        "hip_version": getattr(torch.version, "hip", None),
        "mps": _mps_usable(),
    }
    if _cuda_usable():
        info["gpus"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    return info
