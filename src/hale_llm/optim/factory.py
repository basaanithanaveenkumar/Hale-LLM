"""Build the optimizer named by ``train.optimizer_type``."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from hale_core.nn.optim import build_optimizer as build_hale_core_optimizer
from torch import Tensor, nn

from hale_llm.optim.adamw_optimizer import AdamWOptimizer  # noqa: F401  (registers "adamw")
from hale_llm.optim.muon_optimizer import MuonOptimizer  # noqa: F401  (registers "muon")
from hale_llm.optim.registry import OPTIMIZERS


def build_optimizer(name: str, model: nn.Module | Iterable[Tensor], config: dict[str, Any]):
    """Create an optimizer for ``model`` (a module, or a plain iterable of parameters).

    Optimizers that split parameters by role (``muon``) need the module.

    hale_llm's own optimizers (``adamw``, ``muon``) take precedence over hale_core's
    (``adam``, ``adamw``, ``sgd``) when names clash.
    """
    key = name.lower()
    if not isinstance(model, nn.Module):
        if key in OPTIMIZERS:
            raise TypeError(f"optimizer {key!r} needs the model, not a parameter iterable")
        return build_hale_core_optimizer(key, model, config)
    if key in OPTIMIZERS:
        optimizer_cls = OPTIMIZERS.get(key)
        params = (
            list(model.named_parameters())
            if getattr(optimizer_cls, "needs_named_parameters", False)
            else model.parameters()
        )
        return optimizer_cls(params, config)
    return build_hale_core_optimizer(key, model.parameters(), config)
