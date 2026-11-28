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
    # Activation of the dense `mlp` feed-forward: gelu | relu | silu | swiglu | geglu | reglu (gated ones add a matrix).
    ffn_activation: str = "gelu"
    moe_num_experts: int = 4  # routed experts
    moe_top_k: int = 2  # routed experts used per token
    moe_num_shared: int = 1  # shared experts every token passes through
    moe_hidden_dim: int | None = None  # None = d_ff
    moe_router: str = "noisy_topk"  # topk | noisy_topk (see hale_llm.nn.moe)
    moe_activation: str = "swiglu"  # activation of every expert (see hale_llm.nn.activation)
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

    # Looped transformer (variant `autoregressive_moe_looped`, Parcae, arXiv 2604.12946): a prelude of blocks, a recurrent unit that
    # is run several times, and a coda. The three counts must add up to n_layers (None for all three = thirds, the middle third being looped).
    loop_prelude_layers: int | None = None
    loop_recurrent_layers: int | None = None
    loop_coda_layers: int | None = None
    loop_mean_recurrence: int = 8  # mu_rec: mean number of loops in training (Poisson), and the default at test time
    loop_mean_backprop: int | None = None  # mu_bwd: loops that get a gradient; None = ceil(mu_rec / 2) as in the paper
    loop_sampling: Literal["poisson", "fixed"] = "poisson"  # how the number of loops is drawn per sequence
    loop_eval_recurrence: int | None = None  # loops at validation and generation; None = loop_mean_recurrence
    loop_decay_init: float = 0.5  # initial A_bar (the state's decay per loop), in (0, 1); the paper does not give one
    loop_state_std: float | None = None  # std of the random initial state h_0; None = sqrt(2 / (5 d_model)) ("like-init")
    loop_state_seed: int = 0  # seed of the fixed initial state used at evaluation

    # Offset-conditioned MTP (off by default = the DeepSeek-V3 depths above). With it on, ONE MTP module replaces the `mtp_depth`
    # chained depths, and every training step each sequence gets an offset n in 1..mtp_depth (so `mtp_depth` is the number
    # of offsets, N). At position t the module reads the trunk state h(t), the real token x(t+n) and a learned embedding of n,
    # and predicts x(t+n+1); n = 1 is exactly DeepSeek's depth 1.
    # The offsets are STRATIFIED over the global batch of a step (batch_size x accum_steps x data-parallel ranks): the list
    # 1..N is repeated to fill it, the remainder (G mod N offsets) is added, and the whole list is shuffled, so every offset is
    # used floor(G / N) times per step and a micro-batch of 1 is just one slice of that plan.
    mtp_offset_conditioning: bool = False
    mtp_offset_seed: int | None = None  # seed of the offset order; None = train.seed
    # Which offsets get the extra sequence when G is not a multiple of N: "balanced" = the least-used ones so far (random
    # ties), so the totals over the run stay within 1; "random" = a fresh random subset every step.
    mtp_offset_remainder: Literal["balanced", "random"] = "balanced"
    # Offsets each sequence trains per step (offset mode): 1 = a single offset (default), K = a combination of K distinct
    # offsets, so the module runs K times per sequence. 1 .. mtp_depth.
    mtp_offsets_per_sequence: int = 1
    # True (default): stratified plan over the global batch (every offset and every combination equally often).
    # False: every sequence draws its offsets independently (balanced only in expectation).
    mtp_offset_stratified: bool = True

    # How the losses of the several offsets / depths are combined into the MTP term (which lambda then weights):
    # "sum" adds them up (L_1 + L_2 + ... + L_N, each averaged over its own sequences), "mean" divides by their number.
    # None = the mode's default: "mean" for DeepSeek-V3's depths (as in the paper), "sum" for offset-conditioned MTP.
    mtp_loss_reduction: Literal["sum", "mean"] | None = None

    # Speculative decoding: tokens drafted per round, from 1 to mtp_depth. Every draft costs one pass of the MTP module(s);
    # the trunk then checks all of them in one pass, so the output never depends on this number (it only changes the speed).
    mtp_draft_tokens: int = 1

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
        if self.loop_mean_recurrence < 1 or (self.loop_eval_recurrence is not None and self.loop_eval_recurrence < 1):
            raise ValueError("loop_mean_recurrence and loop_eval_recurrence must be at least 1")
        if self.loop_mean_backprop is not None and self.loop_mean_backprop < 1:
            raise ValueError(f"loop_mean_backprop={self.loop_mean_backprop} must be at least 1")
        if not 0.0 < self.loop_decay_init < 1.0:
            raise ValueError(f"loop_decay_init={self.loop_decay_init} must be strictly between 0 and 1")
        if self.mtp_depth < 1:
            raise ValueError(f"mtp_depth={self.mtp_depth} must be at least 1")
        if self.mtp_loss_weight < 0 or (self.mtp_loss_weight_final is not None and self.mtp_loss_weight_final < 0):
            raise ValueError("mtp_loss_weight and mtp_loss_weight_final cannot be negative")
        if not 0.0 < self.mtp_switch_fraction < 1.0:
            raise ValueError(f"mtp_switch_fraction={self.mtp_switch_fraction} must be between 0 and 1")
        if not 1 <= self.mtp_draft_tokens <= self.mtp_depth:
            raise ValueError(
                f"mtp_draft_tokens={self.mtp_draft_tokens} must be between 1 and mtp_depth={self.mtp_depth}"
            )
        if not 1 <= self.mtp_offsets_per_sequence <= self.mtp_depth:
            raise ValueError(
                f"mtp_offsets_per_sequence={self.mtp_offsets_per_sequence} must be between 1 and mtp_depth={self.mtp_depth}"
            )
        if self.mtp_offset_seed is not None and not 0 <= self.mtp_offset_seed <= 2**31 - 1:
            raise ValueError(f"mtp_offset_seed={self.mtp_offset_seed} must be between 0 and {2**31 - 1}")
        if self.mtp_n_layers < 1:
            raise ValueError(f"mtp_n_layers={self.mtp_n_layers} must be at least 1")
        if self.mtp_d_ff is not None and self.mtp_d_ff < 1:
            raise ValueError(f"mtp_d_ff={self.mtp_d_ff} must be at least 1")
        return self

    def mtp_reduction(self) -> str:
        """``"sum"`` or ``"mean"``: ``mtp_loss_reduction``, or the mode's default when it is None."""
        if self.mtp_loss_reduction is not None:
            return self.mtp_loss_reduction
        return "sum" if self.mtp_offset_conditioning else "mean"

    def mtp_loss_terms(self) -> int:
        """How many per-offset (per-depth) losses the MTP term adds up: ``mtp_depth`` when summed, else 1 (a mean)."""
        return self.mtp_depth if self.mtp_reduction() == "sum" else 1

    def mtp_block_config(self) -> "ModelConfig":
        """The settings the MTP blocks are built with: the trunk's, except where an ``mtp_*`` override is set."""
        update: dict = {}
        if self.mtp_ffn_type is not None:
            update["ffn_type"] = self.mtp_ffn_type
        if self.mtp_d_ff is not None:
            update["d_ff"] = update["moe_hidden_dim"] = self.mtp_d_ff
        return self.model_copy(update=update)
