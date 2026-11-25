"""Name -> feed-forward builder (``model.ffn_type`` in YAML): ``mlp`` or ``moe``."""

from __future__ import annotations

from collections.abc import Callable

from torch import nn

from hale_llm.nn.blocks.feed_forward import FeedForward
from hale_llm.nn.moe import DeepseekMoE
from hale_llm.registry import NamedRegistry

FFNS = NamedRegistry("ffn")


def register_ffn(name: str) -> Callable[[Callable[..., nn.Module]], Callable[..., nn.Module]]:
    """Decorator: register a ``builder(model_cfg) -> nn.Module`` under ``name``."""

    def decorator(builder):
        FFNS.add(name, builder)
        return builder

    return decorator


@register_ffn("mlp")
def build_mlp(model_cfg) -> nn.Module:
    return FeedForward(model_cfg.d_model, model_cfg.d_ff, model_cfg.dropout)


@register_ffn("moe")
def build_moe(model_cfg) -> nn.Module:
    return DeepseekMoE(
        d_model=model_cfg.d_model,
        hidden_dim=model_cfg.moe_hidden_dim or model_cfg.d_ff,
        n_routed_experts=model_cfg.moe_num_experts,
        top_k=model_cfg.moe_top_k,
        n_shared_experts=model_cfg.moe_num_shared,
        router=model_cfg.moe_router,
        activation=model_cfg.moe_activation,
        dropout=model_cfg.dropout,
    )


def build_ffn(name: str, model_cfg) -> nn.Module:
    """Create the feed-forward layer called ``name`` from the model config."""
    return FFNS.get(name)(model_cfg)
