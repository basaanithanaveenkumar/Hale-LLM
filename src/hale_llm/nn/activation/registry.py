"""Name -> activation (``model.ffn_activation`` and ``model.moe_activation`` in YAML)."""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn.functional as F

from hale_llm.nn.activation.activation_spec import ActivationSpec
from hale_llm.registry import NamedRegistry

ACTIVATIONS = NamedRegistry("activation")


def register_activation(
    name: str, fn: Callable[[torch.Tensor], torch.Tensor], gated: bool = False
) -> ActivationSpec:
    """Make ``fn`` selectable from the config under ``name``."""
    return ACTIVATIONS.add(name, ActivationSpec(fn, gated))


def get_activation(name: str) -> ActivationSpec:
    """The spec registered under ``name`` (KeyError lists the valid names)."""
    return ACTIVATIONS.get(name)


# Plain: gelu is GPT-2 / BERT, relu the original, silu (a.k.a. swish) is smooth like gelu.
register_activation("gelu", F.gelu)
register_activation("relu", F.relu)
register_activation("silu", F.silu)
# Gated (GLU family, Shazeer 2020): the name says which function gates. swiglu is Llama / DeepSeek.
register_activation("swiglu", F.silu, gated=True)
register_activation("geglu", F.gelu, gated=True)
register_activation("reglu", F.relu, gated=True)
