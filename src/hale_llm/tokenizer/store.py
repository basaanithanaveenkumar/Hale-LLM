"""Save and load a trained tokenizer next to its checkpoints."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

from loguru import logger
from transformers import PreTrainedTokenizerFast

from hale_llm.tokenizer.special_tokens import EOS_TOKEN


class TokenizerStore:
    """Reads and writes a tokenizer directory (``tokenizer.json`` plus Hugging Face metadata)."""

    _MARKER = "tokenizer.json"

    @classmethod
    def exists(cls, directory: str | Path) -> bool:
        return (Path(directory) / cls._MARKER).is_file()

    @staticmethod
    def wrap(tokenizer_object) -> PreTrainedTokenizerFast:
        """Give a raw ``tokenizers.Tokenizer`` the Hugging Face interface the pipeline expects."""
        return PreTrainedTokenizerFast(
            tokenizer_object=tokenizer_object,
            eos_token=EOS_TOKEN,
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
    def metadata(cls, directory: str | Path) -> dict:
        """What the tokenizer was trained with (kind, dataset, requested vocabulary...); ``{}`` if unknown."""
        path = Path(directory) / "hale_tokenizer.json"
        return json.loads(path.read_text()) if path.is_file() else {}

    @classmethod
    def copy(cls, source: str | Path, destination: str | Path) -> None:
        """Copy a tokenizer folder (atomically, like :meth:`save`) so a run keeps its own copy."""
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        staging = destination.with_name(f"{destination.name}.tmp{os.getpid()}")
        shutil.rmtree(staging, ignore_errors=True)
        shutil.copytree(source, staging)
        try:
            os.rename(staging, destination)
        except OSError:
            shutil.rmtree(staging, ignore_errors=True)
            if not cls.exists(destination):
                raise
        logger.info("copied tokenizer {} -> {}", source, destination)

    @classmethod
    def load(cls, directory: str | Path) -> PreTrainedTokenizerFast:
        if not cls.exists(directory):
            raise FileNotFoundError(f"no tokenizer in {directory}")
        return PreTrainedTokenizerFast.from_pretrained(str(directory))
