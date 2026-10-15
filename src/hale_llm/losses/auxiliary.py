"""Architecture-level auxiliary losses added on top of a variant's task loss.

Task losses (``@register_loss``) only see logits. Some backbones also produce a regulariser
during their forward - the looped transformer's ACT ponder cost ``tau * mean(N + R)`` - which
the trainer must add to the objective. :func:`with_auxiliary_losses` collects them from the
model after the task loss has run its forward, so no variant needs to know about them.
"""

from __future__ import annotations

import torch
from hale_core.nn.stacks import loop_aux_loss, loop_stats
from torch import nn


def auxiliary_losses(model: nn.Module) -> dict[str, torch.Tensor]:
    """Named auxiliary losses of the model's last forward (empty for plain backbones)."""
    out: dict[str, torch.Tensor] = {}
    ponder = loop_aux_loss(model)
    if ponder is not None:
        out["ponder"] = ponder
    return out


def with_auxiliary_losses(
    model: nn.Module, task_loss: torch.Tensor
) -> tuple[torch.Tensor, dict[str, float]]:
    """``task_loss`` plus every auxiliary loss, and the scalars worth logging.

    The scalars hold ``task_loss``, each auxiliary loss and the loop statistics
    (``loop/loops``, ``loop/mean_updates`` ...) - empty extras for a non-looped model.
    """
    extras = auxiliary_losses(model)
    total = task_loss
    for value in extras.values():
        total = total + value
    scalars: dict[str, float] = {}
    if extras:
        scalars["task_loss"] = float(task_loss.detach())
        scalars.update({f"{name}_loss": float(value.detach()) for name, value in extras.items()})
    scalars.update(loop_stats(model))
    return total, scalars
