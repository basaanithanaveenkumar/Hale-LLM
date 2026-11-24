"""Compatibility barrel for older imports (`_is_heading`, ...)."""

from hale_llm.dataloader.loaders import make_dataloader
from hale_llm.dataloader.prompts import (
    collect_dataset_passages,
    dataset_prompt,
    is_heading,
    passage_prompt,
)
from hale_llm.dataloader.sources.huggingface import HuggingFaceTextDataset
from hale_llm.dataloader.sources.overfit import uses_overfit
from hale_llm.dataloader.context_windows import pack_context_windows

_is_heading = is_heading

__all__ = [
    "pack_context_windows",
    "HuggingFaceTextDataset",
    "uses_overfit",
    "collect_dataset_passages",
    "dataset_prompt",
    "passage_prompt",
    "make_dataloader",
    "is_heading",
    "_is_heading",
]
