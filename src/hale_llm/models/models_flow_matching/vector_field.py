"""The linear masking path of the discrete flow-matching variant.

The flow runs from ``t = 0`` (every token masked, the source) to ``t = 1`` (the data). Both the
loss (:mod:`.loss`) and the sampler (:mod:`.sample`) read the path through :func:`mask_prob`, so
a non-linear interpolant is a change to this one function.
"""

from __future__ import annotations

import torch


def mask_prob(t: torch.Tensor | float) -> torch.Tensor | float:
    """Probability that a token is still masked at flow time ``t`` (linear path: ``1 - t``).

    Args:
        t: flow time(s) in ``[0, 1]``.

    Returns:
        ``1 - t``, with the type of ``t``.
    """
    return 1.0 - t
