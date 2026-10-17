"""``model:`` section: backbone architecture and per-variant hyper-parameters."""

from typing import Literal

from hale_core.config.sections.loop import LoopConfig
from pydantic import Field

from hale_llm.config.sections.common import StrictModel


class ModelConfig(StrictModel):
    """Architecture of the shared transformer backbone and the variant-specific knobs.

    ``arch`` picks the HaleBlocks backbone (``default``/``transformer`` GPT stack, ``lgt``
    local-global, ``dit`` time-conditioned, ``looped`` weight-tied). ``attn_type`` must match the
    variant (``causal`` for AR/MTP/looped, ``bidirectional`` for diffusion and flow matching,
    ``block_causal`` for block variants); models validate the combination at construction.
    """

    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 4
    d_ff: int = 512
    dropout: float = 0.0
    max_length: int = 64
    attn_type: Literal["causal", "bidirectional", "block_causal"] = "causal"
    attn_impl: Literal["mha", "gqa", "mqa"] = "mha"
    n_kv_heads: int | None = None
    ffn_type: Literal["mlp", "geglu", "moe"] = "mlp"
    moe_num_experts: int = 4
    moe_top_k: int = 2
    moe_num_shared: int = 1
    use_time_cond: bool = False
    block_size: int | None = None
    n_mtp_heads: int = 2
    # block_flow_matching: kappa schedule of the masking path (linear: kappa = t).
    fm_schedule: Literal["linear", "cosine"] = "linear"
    arch: Literal["default", "transformer", "lgt", "dit", "looped"] = "default"
    sliding_window: int = 512
    local_global_ratio: int = 5
    rope_theta_local: float = 10_000.0
    rope_theta_global: float = 1_000_000.0
    p_rope: float = 0.25
    qk_norm: bool = False
    # arch: looped - n_layers is the shared block, iterated loop.n_loops times (docs/looped.md).
    loop: LoopConfig = Field(default_factory=LoopConfig)
