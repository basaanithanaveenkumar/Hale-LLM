"""Rough parameter count of a model without building it."""

from __future__ import annotations

from hale_llm.config.sections.model import ModelConfig
from hale_llm.nn.activation import get_activation


def estimate_non_embedding_params(model: ModelConfig) -> int:
    """Parameters in the transformer blocks (attention + feed-forward), embeddings excluded.

    Per layer: four ``d x d`` attention projections (q, k, v, output) plus the feed-forward:
    ``2 d d_ff`` for the dense MLP (``3 d d_ff`` if its activation is gated); for a mixture of experts every
    shared and routed expert (``3 d h`` if gated, else ``2 d h``) plus the router. Norms and biases are ignored
    (<1%). The vocabulary is excluded on purpose: it is what this number is used to choose.
    """
    d = model.d_model
    attention = 4 * d * d
    if model.ffn_type == "moe":
        hidden = model.moe_hidden_dim or model.d_ff
        matrices = 3 if get_activation(model.moe_activation).gated else 2
        experts = model.moe_num_experts + model.moe_num_shared
        feed_forward = matrices * d * hidden * experts + d * model.moe_num_experts
    else:
        matrices = 3 if get_activation(model.ffn_activation).gated else 2
        feed_forward = matrices * d * model.d_ff
    return model.n_layers * (attention + feed_forward)
