"""``experiment:`` section: the per-run output folder layout."""

from hale_llm.config.sections.common import StrictModel


class ExperimentConfig(StrictModel):
    """Run folders under ``root/<variant>/<name>`` (checkpoints, logs, TensorBoard, viz, configs).

    ``source_yaml`` is set by :func:`~hale_llm.config.load.load_config`, not by users.
    """

    enabled: bool = True
    root: str = "data/experiments"
    name: str | None = None
    source_yaml: str | None = None
