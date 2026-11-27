"""Build the optimizer named by ``train.optimizer_type``."""

from __future__ import annotations

from typing import Any

from torch import nn

from hale_llm.optim.adam_optimizer import AdamOptimizer  # noqa: F401  (registers "adam")
from hale_llm.optim.adamw_optimizer import AdamWOptimizer  # noqa: F401  (registers "adamw")
from hale_llm.optim.dion_optimizer import DionOptimizer  # noqa: F401  (registers "dion")
from hale_llm.optim.muon_optimizer import MuonOptimizer  # noqa: F401  (registers "muon")
from hale_llm.optim.registry import OPTIMIZERS
from hale_llm.optim.sgd_optimizer import SGDOptimizer  # noqa: F401  (registers "sgd")


def build_optimizer(name: str, model: nn.Module, config: dict[str, Any]):
    """Create the registered optimizer ``name`` for ``model``.

    Optimizers that split parameters by role (``adamw``, ``muon``, ``dion``) set
    ``needs_named_parameters`` and receive ``named_parameters()``; the rest get the
    plain parameter iterator.
    """
    optimizer_cls = OPTIMIZERS.get(name.lower())
    params = (
        list(model.named_parameters())
        if getattr(optimizer_cls, "needs_named_parameters", False)
        else model.parameters()
    )
    return optimizer_cls(params, config)
