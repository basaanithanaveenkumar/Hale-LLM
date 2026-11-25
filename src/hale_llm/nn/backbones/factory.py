"""Build the backbone for a model config."""

from __future__ import annotations

from hale_llm.nn.backbones.transformer_backbone import TransformerBackbone
from hale_llm.nn.norm import NORMS
from hale_llm.nn.positional import POSITIONAL_ENCODINGS


def build_backbone(vocab_size: int, model_cfg, ffn_type: str | None = None) -> TransformerBackbone:
    """Create a :class:`TransformerBackbone`; unknown scheme names fail with the valid choices."""
    _check_name("positional encoding", model_cfg.pos_encoding, POSITIONAL_ENCODINGS)
    _check_name("norm", model_cfg.norm_type, NORMS)
    return TransformerBackbone(vocab_size, model_cfg, ffn_type=ffn_type)


def _check_name(kind: str, name: str, registry) -> None:
    if name not in registry:
        raise KeyError(f"unknown {kind} {name!r}; registered={sorted(registry)}")
