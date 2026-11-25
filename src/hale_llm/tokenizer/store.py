"""Save and load a trained tokenizer next to its checkpoints."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

from loguru import logger
from transformers import PreTrainedTokenizerFast

from hale_llm.tokenizer.special_tokens import EOS_TOKEN, MASK_TOKEN


class TokenizerStore:
    """Reads and writes a tokenizer directory (``tokenizer.json`` plus Hugging Face metadata)."""

    _MARKER = "tokenizer.json"

    @classmethod
    def exists(cls, directory: str | Path) -> bool:
        return (Path(directory) / cls._MARKER).is_file()

    @staticmethod
    def wrap(tokenizer_object, *, with_mask: bool) -> PreTrainedTokenizerFast:
        """Give a raw ``tokenizers.Tokenizer`` the Hugging Face interface the pipeline expects."""
        return PreTrainedTokenizerFast(
            tokenizer_object=tokenizer_object,
            eos_token=EOS_TOKEN,
            mask_token=MASK_TOKEN if with_mask else None,
        )

    @classmethod
    def save(cls, tokenizer: PreTrainedTokenizerFast, directory: str | Path, metadata: dict) -> None:
        directory = Path(directory)
        directory.parent.mkdir(parents=True, exist_ok=True)
        # Write to a private folder, then rename it into place in one step. If several
        # processes (distributed ranks) train at once, the first rename wins and a reader
        # never sees a half-written tokenizer.
        staging = directory.with_name(f"{directory.name}.tmp{os.getpid()}")
        shutil.rmtree(staging, ignore_errors=True)
        staging.mkdir()
        tokenizer.save_pretrained(str(staging))
        (staging / "hale_tokenizer.json").write_text(json.dumps(metadata, indent=2))
        try:
            os.rename(staging, directory)
        except OSError:
            shutil.rmtree(staging, ignore_errors=True)  # another process saved first
            if not cls.exists(directory):
                raise
        logger.info("saved tokenizer to {}", directory)

    @classmethod
    def load(cls, directory: str | Path) -> PreTrainedTokenizerFast:
        if not cls.exists(directory):
            raise FileNotFoundError(f"no tokenizer in {directory}")
        return PreTrainedTokenizerFast.from_pretrained(str(directory))
