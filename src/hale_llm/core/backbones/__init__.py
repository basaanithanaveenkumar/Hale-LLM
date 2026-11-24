# Backbones: hale_core provides lgt/dit and the base class; the transformer
# (GPT) backbone and the factory come from hale_llm.nn so that
# model.pos_encoding and model.norm_type are honoured.
from hale_core.nn.backbones import *  # noqa: F403
from hale_core.nn.backbones import __all__ as _core_all

from hale_llm.nn.backbones import GPTBackbone, build_backbone

TransformerBackbone = GPTBackbone
__all__ = [*_core_all, "GPTBackbone", "TransformerBackbone", "build_backbone"]
