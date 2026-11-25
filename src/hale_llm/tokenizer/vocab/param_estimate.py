"""Rough parameter count of a model without building it."""

from __future__ import annotations

from hale_llm.config.sections.model import ModelConfig


def estimate_non_embedding_params(model: ModelConfig) -> int:
    """Parameters in the transformer blocks (attention + feed-forward), embeddings excluded.

    Per layer: attention projections ``2 d^2 (q, o) + 2 d d_kv (k, v)`` and a feed-forward
    of ``2 d d_ff`` (mlp) or ``3 d d_ff`` (gated). Norms and biases are ignored (<1%).
    A mixture-of-experts FFN counts every expert. The vocabulary is excluded on purpose:
    it is what this number is used to choose.
    """
    d = model.d_model
    if model.attn_impl == "mha":
        kv_heads = model.n_heads
    elif model.attn_impl == "mqa":
        kv_heads = 1
    else:
        kv_heads = model.n_kv_heads or max(1, model.n_heads // 4)
    d_kv = kv_heads * (d // model.n_heads)
    attention = 2 * d * d + 2 * d * d_kv

    matrices = 2 if model.ffn_type == "mlp" else 3
    experts = model.moe_num_experts + model.moe_num_shared if model.ffn_type == "moe" else 1
    feed_forward = matrices * d * model.d_ff * experts

    return model.n_layers * (attention + feed_forward)
