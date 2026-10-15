"""Block flow matching: autoregressive across blocks, discrete flow matching inside a block.

The block analogue of ``flow_matching`` exactly as ``block_diffusion`` is the block analogue of
``diffusion`` - the same clean / noised two-stream layout with block-causal attention, but the
within-block generator is a discrete masking flow (Campbell et al., 2024, arXiv:2402.04997;
Gat et al., 2024, arXiv:2407.15595) integrated from ``t = 0`` (all masked) to ``t = 1`` (data).

path.py     MaskingPath (kappa schedule, rates, loss weights) + per-block corruption
model.py    @register_variant("block_flow_matching")  BlockFlowMatchingLM + collate
loss.py     @register_loss("block_flow_matching")     kappa-weighted masked-token NLL
sample.py   @register_sampler("block_flow_matching")  block-by-block Euler integration
metrics.py  masked_accuracy for this variant
"""

from hale_llm.models.models_block_flow_matching import loss as _loss  # noqa: F401
from hale_llm.models.models_block_flow_matching import metrics as _metrics  # noqa: F401
from hale_llm.models.models_block_flow_matching import sample as _sample  # noqa: F401
from hale_llm.models.models_block_flow_matching.model import (
    BlockFlowMatchingLM,
    collate_block_flow_matching,
)
from hale_llm.models.models_block_flow_matching.path import MaskingPath, corrupt_blocks

__all__ = ["BlockFlowMatchingLM", "MaskingPath", "collate_block_flow_matching", "corrupt_blocks"]
