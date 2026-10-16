"""One inner-epoch view of the d1 log-probability estimate (for :class:`DiffusionLogProbFilm`)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MaskedView:
    """One inner epoch: which prompt positions were masked and the resulting token log-probs."""

    hidden_prompt: tuple[int, ...]
    token_logps: tuple[float, ...]
