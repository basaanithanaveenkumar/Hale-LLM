"""Compose the full run config from section models. No YAML I/O here."""

from loguru import logger
from pydantic import Field, model_validator

from hale_llm.config.sections import (
    DataConfig,
    EvalConfig,
    ExperimentConfig,
    LoggingConfig,
    ModelConfig,
    SampleConfig,
    TrainConfig,
    VizConfig,
)
from hale_llm.config.sections.common import StrictModel


class RunConfig(StrictModel):
    variant: str
    model: ModelConfig = Field(default_factory=ModelConfig)
    train: TrainConfig = Field(default_factory=TrainConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    sample: SampleConfig = Field(default_factory=SampleConfig)
    eval: EvalConfig = Field(default_factory=EvalConfig)
    viz: VizConfig = Field(default_factory=VizConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    experiment: ExperimentConfig = Field(default_factory=ExperimentConfig)
    device: str | None = None

    @model_validator(mode="after")
    def _context_window_sets_model_length(self) -> "RunConfig":
        """``data.context_window`` is the single place the context length is chosen.

        The backbone still reads ``model.max_length``; it is filled in from the data
        config here, so a different ``model.max_length`` is ignored (with a warning).
        """
        window = self.data.context_window
        if window is None:
            return self
        if self.model.max_length != window and "max_length" in self.model.model_fields_set:
            logger.warning(
                "model.max_length={} ignored: the context length comes from data.context_window={}",
                self.model.max_length,
                window,
            )
        self.model.max_length = window
        return self

    def set_context_window(self, tokens: int) -> None:
        """Change the context length after the config was built (keeps data and model in step)."""
        self.data.context_window = tokens
        self.model.max_length = tokens
