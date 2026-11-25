from typing import Literal

from hale_llm.config.sections.common import StrictModel


class TokenizerConfig(StrictModel):
    """Where the tokenizer comes from and how it is trained."""

    # trained    = learn a byte-level BPE from this run's training data (default)
    # pretrained = download `data.tokenizer_name` (e.g. gpt2) from the Hugging Face Hub
    source: Literal["trained", "pretrained"] = "trained"

    # Registered algorithm (see hale_llm.tokenizer.training): superbpe | bpe.
    kind: str = "superbpe"

    # None = choose from the model size with `vocab_policy`; or an explicit number.
    vocab_size: int | None = None
    # Registered rule (see hale_llm.tokenizer.vocab): by_model_size.
    vocab_policy: str = "by_model_size"

    # SuperBPE: share of the vocabulary learned with word boundaries (stage 1); the rest is
    # learned across words (stage 2). 0.8 follows the SuperBPE paper's regime.
    transition_fraction: float = 0.8
    # SuperBPE stage 2 runs over at most this many tokens of training text (memory/time cap).
    stage2_max_tokens: int = 2_000_000
    # A merge is learned only if its pair occurs at least this often.
    min_frequency: int = 2

    # Where the trained tokenizer is stored. Experiment runs rewrite this to <run>/tokenizer.
    path: str = "data/tokenizer"
