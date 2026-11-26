from typing import Literal

from hale_llm.config.sections.common import StrictModel


class LeanConfig(StrictModel):
    """Settings of the lean single-file trainer (``train.trainer: lean``)."""

    accum_steps: int = 1  # micro-batches per optimizer step; effective batch = batch_size * this
    warmup_steps: int = 50  # lr ramps linearly from ~0 to its base value
    warmdown_frac: float = 0.2  # last fraction of steps in which lr decays linearly to 0
    eval_every: int = 250  # validation every N steps; 0 disables
    eval_batches: int = 20  # validation batches per evaluation
    keep_best: int = 5  # keep this many best checkpoints in <run>/checkpoints/best; 0 disables
    best_by: Literal["loss", "token_accuracy"] = "loss"  # validation score that ranks them
    compile: bool = False  # wrap the model in torch.compile
