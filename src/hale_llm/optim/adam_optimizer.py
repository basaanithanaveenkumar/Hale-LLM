"""Plain Adam."""

from __future__ import annotations

from typing import Any

import torch

from hale_llm.optim.base_optimizer import BaseOptimizer
from hale_llm.optim.registry import register_optimizer


@register_optimizer("adam")
class AdamOptimizer(BaseOptimizer):
    """``torch.optim.Adam`` with ``lr`` and ``weight_decay`` from the config."""

    def _make_optimizer(self, params: Any, config: dict[str, Any]):
        return torch.optim.Adam(
            params, lr=config.get("lr", 1e-3), weight_decay=config.get("weight_decay", 0.0)
        )
