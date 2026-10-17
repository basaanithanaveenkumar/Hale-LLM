"""Block diffusion: autoregressive across blocks, absorbing-state diffusion inside a block.

``model.py`` holds the variant and its collate function, ``loss.py`` the registered loss and
``sample.py`` the registered sampler. Importing the package registers all of them.
"""

from hale_llm.models.models_block_diffusion import metrics as _metrics  # noqa: F401
from hale_llm.models.models_block_diffusion.loss import block_diffusion_loss
from hale_llm.models.models_block_diffusion.model import BlockDiffusionLM, collate_block_diffusion
from hale_llm.models.models_block_diffusion.sample import sample_block_diffusion

__all__ = [
    "BlockDiffusionLM",
    "block_diffusion_loss",
    "collate_block_diffusion",
    "sample_block_diffusion",
]
