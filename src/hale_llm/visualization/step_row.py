"""One denoising step rendered as a row of a paper-style figure."""

from __future__ import annotations

from dataclasses import dataclass, field

from hale_llm.visualization.token_span import TokenSpan


@dataclass
class StepRow:
    """One figure row: the step label and its coloured :class:`TokenSpan` pieces."""

    tokens: list[TokenSpan] = field(default_factory=list)
    # kept so older tests that pass text= still construct
    text: str = ""
    spans: list[tuple[int, int]] = field(default_factory=list)
