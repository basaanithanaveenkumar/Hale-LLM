"""Discrete flow matching on a linear masking path (``t = 0`` all masks → ``t = 1`` data).

``model.py`` holds the variant and its collate function, ``loss.py`` the registered loss,
``sample.py`` the registered sampler and ``vector_field.py`` the masking path they share.
Importing the package registers all of them.
"""

from hale_llm.models.models_flow_matching import metrics as _metrics  # noqa: F401
from hale_llm.models.models_flow_matching.loss import flow_matching_loss
from hale_llm.models.models_flow_matching.model import FlowMatchingLM, collate_flow_matching
from hale_llm.models.models_flow_matching.sample import sample_flow_matching

__all__ = ["FlowMatchingLM", "collate_flow_matching", "flow_matching_loss", "sample_flow_matching"]
