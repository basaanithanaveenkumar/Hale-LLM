"""Masked diffusion (MDLM): absorbing-state corruption with a pluggable noise schedule.

``model.py`` holds the variant and its collate function, ``loss.py`` the registered loss,
``sample.py`` the registered sampler, ``noise_schedule.py`` / ``log_linear_schedule.py`` the
forward-masking strategy. Importing the package registers all of them.
"""

from hale_llm.models.models_diffusion import metrics as _metrics  # noqa: F401
from hale_llm.models.models_diffusion.log_linear_schedule import LogLinearSchedule
from hale_llm.models.models_diffusion.loss import diffusion_loss
from hale_llm.models.models_diffusion.model import MaskedDiffusionLM, collate_diffusion
from hale_llm.models.models_diffusion.noise_schedule import NoiseSchedule
from hale_llm.models.models_diffusion.sample import sample_diffusion

__all__ = [
    "LogLinearSchedule",
    "MaskedDiffusionLM",
    "NoiseSchedule",
    "collate_diffusion",
    "diffusion_loss",
    "sample_diffusion",
]
