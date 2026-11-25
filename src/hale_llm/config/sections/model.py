from typing import Literal

from pydantic import model_validator

from hale_llm.config.sections.common import StrictModel


class ModelConfig(StrictModel):
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 4
    # Hidden width of the feed-forward network (and of each MoE expert unless moe_hidden_dim).
    d_ff: int = 512
    dropout: float = 0.0
    max_length: int = 64
    # Share the token embedding with the output head (saves vocab_size * d_model parameters).
    tie_embeddings: bool = True

    # Feed-forward network: a dense MLP, or the DeepSeek-style mixture of experts.
    ffn_type: Literal["mlp", "moe"] = "mlp"
    moe_num_experts: int = 4  # routed experts
    moe_top_k: int = 2  # routed experts used per token
    moe_num_shared: int = 1  # shared experts every token passes through
    moe_hidden_dim: int | None = None  # None = d_ff
    moe_router: str = "noisy_topk"  # topk | noisy_topk (see hale_llm.nn.moe)
    moe_activation: Literal["swiglu", "gelu"] = "swiglu"
    # Weight of the load-balancing loss that keeps tokens spread over the experts.
    moe_aux_loss_coef: float = 0.01

    # Name of a registered positional encoding (see hale_llm.nn.positional): rope | learned.
    pos_encoding: str = "rope"
    # Name of a registered norm layer (see hale_llm.nn.norm): layer | rms.
    norm_type: str = "layer"
    rope_theta: float = 10_000.0
    # Stretch RoPE beyond the trained length (see hale_llm.nn.positional.scaling):
    # none | linear | yarn. `rope_scaling_factor` = target length / trained length;
    # `rope_original_length` is the length the weights were trained at.
    rope_scaling: str = "none"
    rope_scaling_factor: float = 1.0
    rope_original_length: int = 4096

    @model_validator(mode="after")
    def _check_shapes(self) -> "ModelConfig":
        if self.d_model % self.n_heads != 0:
            raise ValueError(f"d_model={self.d_model} must be divisible by n_heads={self.n_heads}")
        if self.moe_top_k > self.moe_num_experts:
            raise ValueError(
                f"moe_top_k={self.moe_top_k} cannot exceed moe_num_experts={self.moe_num_experts}"
            )
        return self
