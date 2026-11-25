"""SGD with momentum."""

from __future__ import annotations

from typing import Any

import torch

from hale_llm.optim.base_optimizer import BaseOptimizer
from hale_llm.optim.registry import register_optimizer


@register_optimizer("sgd")
class SGDOptimizer(BaseOptimizer):
    """``torch.optim.SGD``; reads ``lr``, ``momentum``, ``weight_decay``, ``nesterov``."""

    def _make_optimizer(self, params: Any, config: dict[str, Any]):
        return torch.optim.SGD(
            params,
            lr=config.get("lr", 1e-3),
            momentum=config.get("momentum", 0.9),
            weight_decay=config.get("weight_decay", 0.0),
            nesterov=config.get("nesterov", False),
        )
