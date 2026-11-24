from typing import Literal

from hale_core.config.sections.webdataset import WebDatasetConfig
from pydantic import Field

from hale_llm.config.sections.common import StrictModel


class DataConfig(StrictModel):
    source: Literal["huggingface", "overfit"] = "huggingface"
    # TinyStories (MIT): short children's stories, easy for small models to fit.
    dataset: str = "roneneldan/TinyStories"
    subset: str | None = None
    train_split: str = "train"
    val_split: str = "validation"
    text_field: str = "text"
    cache_dir: str | None = None
    tokenizer_name: str = "gpt2"
    train_size: int | None = None
    val_size: int | None = None
    overfit_text: str | None = None
    n_overfit_copies: int = 64
    add_special_tokens: bool = False
    # Tokens per training example, and therefore the model's context length. Documents are
    # joined with EOS between them and cut into back-to-back windows of this size.
    # None = use model.max_length.
    context_window: int | None = None

    # Convert the tokenised windows to WebDataset tar shards once and stream them afterwards
    # (enabled: true); off by default.
    webdataset: WebDatasetConfig = Field(default_factory=WebDatasetConfig)
