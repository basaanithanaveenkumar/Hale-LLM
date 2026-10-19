"""Small built-in English corpus tokenized offline for post-training-quantization calibration."""

from __future__ import annotations

from collections.abc import Iterator

import torch
from loguru import logger

#: Varied prose (narrative, technical, dialogue, lists) so activation ranges are realistic.
PARAGRAPHS = (
    "The river ran high that spring, and the old ferryman counted the days until the water fell "
    "back "
    "between its banks. Travellers waited on both shores, trading news and bread while the "
    "current roared.",
    "A transformer layer applies self-attention followed by a position-wise feed-forward network; "
    "residual "
    "connections and normalization keep the activations well scaled across dozens of layers.",
    '"Are you sure the bridge will "
     "hold?" she asked. "It held last winter," he said, "and the winter '
    'before that. Bridges remember how to stand if you let them."',
    "Ingredients: two cups of flour, one teaspoon of salt, three eggs, a cup of milk. Whisk until "
    "smooth, "
    "rest for thirty minutes, then cook in a hot pan for one minute on each side.",
    "In 1969 the first humans walked on the Moon. The mission lasted eight days and returned 21.5 "
    "kilograms "
    "of lunar samples, which scientists still study to understand the early solar system.",
    "Quantization maps floating-point values to a small set of integers. Calibration chooses the "
    "scale of "
    "that mapping so the most frequent values keep their precision while rare outliers are "
    "clipped.",
)


class CalibrationCorpus:
    """Tokenize :data:`PARAGRAPHS` with the run's tokenizer, if it is available offline.

    Calibration must never trigger a network download in a deployment job, so the tokenizer is
    loaded with ``local_files_only=True``; when it is not cached the adapter returns ``None``
    and the pipeline falls back to random calibration batches.
    """

    def __init__(self, tokenizer_name: str) -> None:
        self.tokenizer_name = tokenizer_name

    def tokenizer(self):
        """The run's tokenizer loaded from the local cache only, or ``None`` (never downloads)."""
        try:
            from transformers import AutoTokenizer

            tok = AutoTokenizer.from_pretrained(self.tokenizer_name, local_files_only=True)
        except Exception as exc:  # noqa: BLE001 - missing cache, missing package, bad name
            logger.info(
                "tokenizer {} not available offline ({}); calibration will be random",
                self.tokenizer_name,
                type(exc).__name__,
            )
            return None
        if tok.mask_token is None:  # match hale_llm.dataloader.tokenizer.setup_tokenizer
            tok.add_special_tokens({"mask_token": "[MASK]"})
        return tok

    def token_windows(
        self, seq_len: int, num_batches: int, vocab_size: int
    ) -> Iterator[torch.Tensor] | None:
        """``num_batches`` tensors ``(1, seq_len)`` cut from the tokenized corpus (cycled)."""
        tok = self.tokenizer()
        if tok is None:
            return None
        ids = tok(" ".join(PARAGRAPHS), add_special_tokens=False)["input_ids"]
        ids = [i for i in ids if i < vocab_size]
        if len(ids) < seq_len:
            ids = ids * (seq_len // max(len(ids), 1) + 1)
        stride = max(1, (len(ids) - seq_len) // max(num_batches, 1))

        def windows() -> Iterator[torch.Tensor]:
            for k in range(num_batches):
                start = (k * stride) % max(1, len(ids) - seq_len + 1)
                yield torch.tensor([ids[start : start + seq_len]], dtype=torch.long)

        return windows()
