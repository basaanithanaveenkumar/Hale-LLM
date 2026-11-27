"""The load-balancing loss of every mixture-of-experts layer inside a module."""

from __future__ import annotations

import torch
from torch import nn

from hale_llm.nn.moe.deepseek_moe import DeepseekMoE


def auxiliary_loss(module: nn.Module) -> torch.Tensor:
    """Sum of the balancing loss of all :class:`DeepseekMoE` layers in ``module`` (zero if there are none).

    Each layer sets its loss on every forward pass (zero outside training), so call this after the forward
    pass. Works on a block, the trunk, or a whole model alike: one function instead of a method per class.
    """
    losses = [
        layer.aux_loss
        for layer in module.modules()
        if isinstance(layer, DeepseekMoE) and layer.aux_loss is not None
    ]
    if not losses:
        parameter = next(module.parameters(), None)
        return torch.zeros((), device=parameter.device if parameter is not None else None)
    return torch.stack(losses).sum()
