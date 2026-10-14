"""Network-free tokenizer for Ray training tests (same behaviour as conftest.TinyTokenizer)."""

from __future__ import annotations

import torch


class TinyTokenizer:
    pad_token_id = 0
    eos_token = "<eos>"
    mask_token_id = 1
    vocab_size = 64

    def __len__(self) -> int:
        return self.vocab_size

    def __call__(
        self,
        text,
        truncation=True,
        padding="max_length",
        max_length=64,
        return_tensors="pt",
        add_special_tokens=False,
    ):
        ids = [2 + (ord(c) % (self.vocab_size - 2)) for c in text]
        if truncation:
            ids = ids[:max_length]
        if padding == "max_length":
            ids = ids + [self.pad_token_id] * (max_length - len(ids))
        t = torch.tensor(ids, dtype=torch.long)
        return {"input_ids": t.unsqueeze(0)} if return_tensors == "pt" else {"input_ids": ids}

    def decode(self, ids, skip_special_tokens=True) -> str:
        return "".join(chr(32 + (i % 95)) for i in ids if not (skip_special_tokens and i in (0, 1)))
