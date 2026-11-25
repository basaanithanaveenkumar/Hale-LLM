from typing import Literal

from hale_llm.config.sections.common import StrictModel


class LoggingConfig(StrictModel):
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    log_file: str | None = "logs/hale_llm.log"
    
    # TensorBoard configuration
    tensorboard: bool = True
    tensorboard_dir: str = "runs"
