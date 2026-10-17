"""One deployment adapter per HaloDiffusionLLM variant (registered in ``LLM_ADAPTERS``)."""

from hale_llm.deploy.adapters.autoregressive_lm_adapter import AutoregressiveLMAdapter
from hale_llm.deploy.adapters.block_diffusion_lm_adapter import BlockDiffusionLMAdapter
from hale_llm.deploy.adapters.block_flow_matching_lm_adapter import BlockFlowMatchingLMAdapter
from hale_llm.deploy.adapters.calibration_corpus import CalibrationCorpus
from hale_llm.deploy.adapters.flow_matching_lm_adapter import FlowMatchingLMAdapter
from hale_llm.deploy.adapters.hale_llm_adapter import HaleLLMAdapter
from hale_llm.deploy.adapters.looped_lm_adapter import LoopedLMAdapter
from hale_llm.deploy.adapters.masked_diffusion_lm_adapter import MaskedDiffusionLMAdapter
from hale_llm.deploy.adapters.multi_token_prediction_lm_adapter import MultiTokenPredictionLMAdapter
from hale_llm.deploy.adapters.run_config_resolver import RunConfigResolver

__all__ = [
    "AutoregressiveLMAdapter",
    "BlockDiffusionLMAdapter",
    "BlockFlowMatchingLMAdapter",
    "CalibrationCorpus",
    "FlowMatchingLMAdapter",
    "HaleLLMAdapter",
    "LoopedLMAdapter",
    "MaskedDiffusionLMAdapter",
    "MultiTokenPredictionLMAdapter",
    "RunConfigResolver",
]
