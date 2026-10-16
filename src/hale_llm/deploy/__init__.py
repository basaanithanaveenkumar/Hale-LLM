"""Deployment of HaloDiffusionLLM variants to NVIDIA, Apple, AMD and Qualcomm targets.

Importing this package registers one adapter per variant in the ``hale_llm`` scope of the
HaleBlocks adapter registry; deployment configs reference them as ``hale_llm.<Adapter>`` and
list ``custom_imports: [hale_llm.deploy]``. See ``docs/deploy.md``.
"""

from hale_llm.deploy.adapters import (
    AutoregressiveLMAdapter,
    BlockDiffusionLMAdapter,
    BlockFlowMatchingLMAdapter,
    FlowMatchingLMAdapter,
    HaleLLMAdapter,
    LoopedLMAdapter,
    MaskedDiffusionLMAdapter,
    MultiTokenPredictionLMAdapter,
)
from hale_llm.deploy.registry import LLM_ADAPTERS

__all__ = [
    "LLM_ADAPTERS", "AutoregressiveLMAdapter", "BlockDiffusionLMAdapter", "BlockFlowMatchingLMAdapter",
    "FlowMatchingLMAdapter", "HaleLLMAdapter", "LoopedLMAdapter", "MaskedDiffusionLMAdapter",
    "MultiTokenPredictionLMAdapter",
]
