"""Character tokenizer for the explainer animations: one printable character = one token.

Explainers train tiny models on one sentence and draw every token as a box, so a readable,
reversible vocabulary matters more than compression. The interface is the subset of a
Hugging Face tokenizer that the variants, samplers and collators use.
"""

from __future__ import annotations

import string

import torch

from hale_llm.exceptions import VisualizationError

PAD, MASK, EOS = "<pad>", "<mask>", "<eos>"


class CharTokenizer:
    """``<pad>`` = 0, ``<mask>`` = 1, ``<eos>`` = 2, then the printable ASCII characters."""

    def __init__(self, alphabet: str = string.printable[:95]) -> None:
        self.specials = [PAD, MASK, EOS]
        self.chars = list(dict.fromkeys(alphabet))
        self.itos = self.specials + self.chars
        self.stoi = {c: i for i, c in enumerate(self.itos)}
        self.pad_token, self.mask_token, self.eos_token = PAD, MASK, EOS
        self.pad_token_id, self.mask_token_id, self.eos_token_id = 0, 1, 2

    @property
    def vocab_size(self) -> int:
        """Specials plus printable characters."""
        return len(self.itos)

    def __len__(self) -> int:
        return self.vocab_size

    def encode(self, text: str) -> list[int]:
        """Character ids of ``text`` (raises ``VisualizationError`` for characters outside the
        alphabet).
        """
        unknown = sorted({c for c in text if c not in self.stoi})
        if unknown:
            raise VisualizationError(f"characters outside the vocabulary: {unknown}")
        return [self.stoi[c] for c in text]

    def __call__(
        self,
        text: str,
        truncation: bool = True,
        padding: bool | str = False,
        max_length: int | None = None,
        return_tensors: str | None = None,
        add_special_tokens: bool = False,
    ) -> dict:
        ids = self.encode(text)
        if truncation and max_length is not None:
            ids = ids[:max_length]
        if padding == "max_length" and max_length is not None:
            ids = ids + [self.pad_token_id] * (max_length - len(ids))
        if return_tensors == "pt":
            return {"input_ids": torch.tensor([ids], dtype=torch.long)}
        return {"input_ids": ids}

    def token(self, token_id: int) -> str:
        """Display string of one id (``<mask>`` etc. for specials)."""
        return self.itos[int(token_id)]

    def decode(self, ids, skip_special_tokens: bool = True) -> str:
        """Characters of ``ids``, skipping the special tokens by default."""
        ids = ids.tolist() if torch.is_tensor(ids) else list(ids)
        return "".join(
            self.itos[i] for i in ids if not (skip_special_tokens and i < len(self.specials))
        )
