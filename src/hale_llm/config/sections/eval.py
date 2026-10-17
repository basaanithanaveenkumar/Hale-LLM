"""``eval:`` section: which metrics run and how often."""

from pydantic import Field

from hale_llm.config.sections.common import StrictModel


class EvalConfig(StrictModel):
    """Registered metric names to compute, evaluation frequency and an optional batch cap."""
    metrics: list[str] = Field(
        default_factory=lambda: [
            "loss",
            "perplexity",
            "bits_per_token",
            "token_accuracy",
            "masked_accuracy",
        ]
    )
    every_n_epochs: int | None = 1
    max_batches: int | None = None
