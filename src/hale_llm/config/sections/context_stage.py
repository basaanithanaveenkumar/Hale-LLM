from hale_llm.config.sections.common import StrictModel


class ContextStageConfig(StrictModel):
    """One later stage of a context-extension schedule (``train.context_stages``).

    Training starts at ``data.context_window`` and ``model.rope_theta``. From
    ``start_fraction`` of the total steps on, the stage switches to its own context
    window and RoPE base. The batch is shrunk so every step still sees about the same
    number of tokens.
    """

    name: str = "extend"
    # Fraction of total steps at which this stage begins, in (0, 1). 0.95 = last 5%.
    start_fraction: float
    # Tokens per window in this stage (must exceed the previous stage's).
    context_window: int
    # Larger RoPE base for the longer context (1M-5M is typical for 32k).
    rope_theta: float
    # Share of this stage's windows taken from naturally long documents; the rest is
    # the normal mix, so short-context quality does not regress. 0 = no long-document mixing.
    long_doc_fraction: float = 0.5
    # A document counts as "long" if it has at least this many tokens
    # (None = half of this stage's context window).
    long_doc_min_tokens: int | None = None
