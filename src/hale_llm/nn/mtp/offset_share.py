"""One micro-batch's share of the offsets of a training step."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class OffsetShare:
    """What :class:`StratifiedOffsetAssigner` hands to one rank for one micro-batch.

    Attributes:
        offsets: ``[B]`` the offset of each sequence of this micro-batch, values in ``1 .. n_offsets``.
        global_counts: ``[n_offsets]`` how many sequences of the WHOLE step (all micro-batches, all ranks) got each
            offset; entry ``n - 1`` is the count of offset ``n``. The loss divides by it so that offsets add up.
        micro_batches: how many micro-batches make up the step (``accum_steps x data-parallel ranks``). The trainer
            averages the micro-batch losses over them, so the loss multiplies by this number to undo that average.
    """

    offsets: torch.Tensor
    global_counts: torch.Tensor
    micro_batches: int
