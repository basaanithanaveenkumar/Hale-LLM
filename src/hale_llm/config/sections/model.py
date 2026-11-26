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

    # Packed windows hold several documents (separated by the end-of-text token). With document_masking a token
    # only attends to tokens of its own document, as in Llama 3 and SmolLM3 (DeepSeek-V3 does not mask). Off by
    # default: GPT-2 style training lets all documents of a window see each other. Costs a [B, S, S] mask per step.
    document_masking: bool = False

    # Multi-token prediction (DeepSeek-V3 style), used by the `autoregressive_mtp` variant.
    #
    # How many tokens are predicted at once: at every position i the main head predicts the next token
    # t_{i+1}, and each MTP depth k = 1..mtp_depth predicts one token further ahead, t_{i+k+1}. With the
    # default mtp_depth = 1 each position predicts 1 + 1 = 2 tokens: t+1 (main head) and t+2 (MTP depth 1).
    # mtp_depth = 4 would predict 5: t+1 (main) and t+2, t+3, t+4, t+5 (depths 1-4). The extra depths cost
    # training compute and parameters and are only needed for training and for speculative decoding.
    mtp_depth: int = 1  # extra tokens predicted per position (1 -> the model predicts 2 tokens at a time)

    # Loss: total = next_token + lambda * (1 / mtp_depth) * sum(loss of each depth). Lambda follows a step
    # schedule like DeepSeek-V3 (0.3 for the first 10T of 14.8T tokens, then 0.1): `mtp_loss_weight` until
    # `mtp_switch_fraction` of the training steps, `mtp_loss_weight_final` after. The 1 / mtp_depth factor keeps
    # the total MTP weight comparable between runs with different depths. Set `mtp_loss_weight_final: null`
    # for a constant lambda.
    mtp_loss_weight: float = 0.3
    mtp_loss_weight_final: float | None = 0.1
    mtp_switch_fraction: float = 0.68  # 10 / 14.8 in DeepSeek-V3

    # The transformer block(s) inside each MTP depth. The width and the number of heads are the trunk's (the depth
    # reads the trunk's hidden states and shares its rotary positions); the rest defaults to the trunk's values, as
    # when None. DeepSeek-V3 uses one block per depth.
    mtp_n_layers: int = 1  # blocks per MTP depth
    mtp_d_ff: int | None = None  # None = d_ff (the expert width for a mixture of experts)
    mtp_ffn_type: Literal["mlp", "moe"] | None = None  # None = ffn_type

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
        if self.mtp_depth < 1:
            raise ValueError(f"mtp_depth={self.mtp_depth} must be at least 1")
        if self.mtp_loss_weight < 0 or (self.mtp_loss_weight_final is not None and self.mtp_loss_weight_final < 0):
            raise ValueError("mtp_loss_weight and mtp_loss_weight_final cannot be negative")
        if not 0.0 < self.mtp_switch_fraction < 1.0:
            raise ValueError(f"mtp_switch_fraction={self.mtp_switch_fraction} must be between 0 and 1")
        if self.mtp_n_layers < 1:
            raise ValueError(f"mtp_n_layers={self.mtp_n_layers} must be at least 1")
        if self.mtp_d_ff is not None and self.mtp_d_ff < 1:
            raise ValueError(f"mtp_d_ff={self.mtp_d_ff} must be at least 1")
        return self

    def mtp_block_config(self) -> "ModelConfig":
        """The settings the MTP blocks are built with: the trunk's, except where an ``mtp_*`` override is set."""
        update: dict = {}
        if self.mtp_ffn_type is not None:
            update["ffn_type"] = self.mtp_ffn_type
        if self.mtp_d_ff is not None:
            update["d_ff"] = update["moe_hidden_dim"] = self.mtp_d_ff
        return self.model_copy(update=update)
