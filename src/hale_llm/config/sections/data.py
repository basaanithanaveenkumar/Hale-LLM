"""``data:`` section: dataset source, tokenizer and windowing."""

from typing import Literal

from hale_core.config.sections.webdataset import WebDatasetConfig
from pydantic import Field

from hale_llm.config.sections.common import StrictModel


class DataConfig(StrictModel):
    """Where training text comes from and how it is cut into fixed-length windows.

    ``source: huggingface`` streams ``dataset``/``subset`` from the Hub; ``source: overfit`` repeats
    ``overfit_text`` ``n_overfit_copies`` times (sanity runs). Text is split into words and packed
    into ``model.max_length``-token windows that start every ``stride_words`` words.
    """
    source: Literal["huggingface", "overfit"] = "huggingface"
    dataset: str = "Salesforce/wikitext"
    subset: str | None = "wikitext-2-raw-v1"
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
    stride_words: int = 10

    # Convert the tokenised windows to WebDataset tar shards once and stream them afterwards
    # (enabled: true); off by default.
    webdataset: WebDatasetConfig = Field(default_factory=WebDatasetConfig)
