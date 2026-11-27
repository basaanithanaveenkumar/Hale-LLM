from typing import Literal

from pydantic import Field

from hale_llm.config.sections.common import StrictModel
from hale_llm.config.sections.context_stage import ContextStageConfig
from hale_llm.config.sections.lean import LeanConfig
from hale_llm.config.sections.titan import TitanConfig


class TrainConfig(StrictModel):
    # Which training backend runs: lean (default, single-file loop) | torchtitan (NVIDIA GPUs
    # only). Names come from the backend registry.
    trainer: str = "lean"
    lean: LeanConfig = Field(default_factory=LeanConfig)
    # Optional context-extension schedule (lean trainer, RoPE transformer only). Empty =
    # train at data.context_window throughout. See docs/context_length.md.
    context_stages: list[ContextStageConfig] = Field(default_factory=list)
    titan: TitanConfig = Field(default_factory=TitanConfig)
    steps: int | None = 300
    epochs: int | None = None
    lr: float = 1e-3
    batch_size: int = 8
    weight_decay: float = 0.0
    log_every: int = 50
    grad_clip: float = 1.0
    seed: int = 0
    checkpoint_path: str = "checkpoints/last.pt"
    checkpoint_every_epoch: bool = True
    resume: bool = True
    loss_type: Literal["ce", "focal", "label_smoothing"] = "ce"
    focal_gamma: float = 2.0
    focal_alpha: float = 1.0
    # Registered optimizer name: adamw | adam | sgd | muon | dion.
    # Default adamw = LLM recipe: betas (adam_beta1, adam_beta2), decay on weight matrices only.
    optimizer_type: str = "adamw"
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    # Muon only: learning rate and momentum for the hidden weight matrices
    # (everything else uses `lr` with AdamW). Typical Muon lr is ~0.02.
    muon_lr: float = 0.02
    muon_momentum: float = 0.95
    # Dion only (optimizer_type: dion): matrices use dion_lr, momentum dion_mu and a power-iteration rank of at most
    # dion_rank (capped at the smaller side of each matrix); everything else uses `lr` with AdamW.
    dion_lr: float = 0.02
    dion_mu: float = 0.95
    dion_rank: int = 768

