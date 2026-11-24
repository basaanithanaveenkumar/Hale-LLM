"""Build the backbone named by ``model.arch``.

``transformer`` / ``default`` resolve to this package's :class:`GPTBackbone`
(configurable positional encoding and norm). Other architectures
(``lgt``, ``dit``) are built by hale_core and keep their own fixed design.
"""

from __future__ import annotations

from hale_core.nn.backbones.factory import BACKBONES as HALE_CORE_BACKBONES
from hale_core.nn.backbones.factory import backbone_kwargs
from loguru import logger

from hale_llm.nn.backbones.gpt_backbone import GPTBackbone
from hale_llm.nn.norm import NORMS
from hale_llm.nn.positional import POSITIONAL_ENCODINGS

_OWN_ARCHS = {"default": GPTBackbone, "transformer": GPTBackbone}


def build_backbone(vocab_size: int, model_cfg=None, **overrides):
    """Create a backbone from a ``ModelConfig`` (or plain keyword arguments)."""
    if model_cfg is not None:
        kwargs = backbone_kwargs(vocab_size, model_cfg, **overrides)
        kwargs["pos_encoding"] = model_cfg.pos_encoding
        kwargs["norm_type"] = model_cfg.norm_type
        kwargs["rope_theta"] = model_cfg.rope_theta
        kwargs.update(overrides)
    else:
        kwargs = {"vocab_size": vocab_size, **overrides}
        kwargs.setdefault("arch", "transformer")

    arch = kwargs["arch"]
    if arch in _OWN_ARCHS:
        _check_name("positional encoding", kwargs.get("pos_encoding", "rope"), POSITIONAL_ENCODINGS)
        _check_name("norm", kwargs.get("norm_type", "layer"), NORMS)
        return _OWN_ARCHS[arch](**kwargs)

    # lgt and dit fix their own position handling (RoPE) and norm (RMSNorm).
    logger.info(
        "arch={}: positional encoding and norm are fixed by the architecture; "
        "model.pos_encoding / model.norm_type are ignored",
        arch,
    )
    return HALE_CORE_BACKBONES.get(arch)(**kwargs)


def _check_name(kind: str, name: str, registry) -> None:
    """Fail early, with the list of valid names, when the config names an unknown scheme."""
    if name not in registry:
        raise KeyError(f"unknown {kind} {name!r}; registered={sorted(registry)}")
