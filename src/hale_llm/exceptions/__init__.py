"""Exception hierarchy of :mod:`hale_llm`.

.. code-block:: text

    HaleCoreError                    (hale_core.exceptions; also ConfigError, RegistryError)
    └── HaleLLMError
        ├── ModelConfigurationError  (also ValueError)
        ├── DataPipelineError        (also ValueError)
        ├── GenerationError          (also ValueError)
        ├── VisualizationError       (also ValueError)
        ├── ExperimentNotFoundError  (also FileNotFoundError)
        ├── TrainingError            (also RuntimeError)
        │   └── NonFiniteLossError   (also FloatingPointError)
        └── OnlineRLError            (hale_llm.rl; also RuntimeError)

Each class also derives from the built-in exception the code raised before the hierarchy
existed, so existing ``except ValueError`` (etc.) call sites keep working.
"""

from hale_llm.exceptions.data_pipeline_error import DataPipelineError
from hale_llm.exceptions.experiment_not_found_error import ExperimentNotFoundError
from hale_llm.exceptions.generation_error import GenerationError
from hale_llm.exceptions.hale_llm_error import HaleLLMError
from hale_llm.exceptions.model_configuration_error import ModelConfigurationError
from hale_llm.exceptions.non_finite_loss_error import NonFiniteLossError
from hale_llm.exceptions.training_error import TrainingError
from hale_llm.exceptions.visualization_error import VisualizationError

__all__ = ["DataPipelineError", "ExperimentNotFoundError", "GenerationError", "HaleLLMError", "ModelConfigurationError", "NonFiniteLossError", "TrainingError", "VisualizationError"]
