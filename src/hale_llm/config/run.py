"""Compose the full run config from section models. No YAML I/O here."""

from pydantic import Field

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
    """The complete, validated configuration of one hale_llm run.

    Unknown keys are rejected at every level (``extra="forbid"``), so a typo in YAML fails at load
    time instead of silently using a default. ``variant`` selects the registered model, loss,
    sampler, collate function and metrics; ``device=None`` picks CUDA, then MPS, then CPU.
    """
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
