"""Is the tokenizer right? Round trips, and an independent byte-level BPE written from the algorithm to compare against."""

from __future__ import annotations

import json

import pytest
import regex

from hale_llm.config.sections.tokenizer import TokenizerConfig
from hale_llm.tokenizer import TokenizerStore
from hale_llm.tokenizer.pretokenizer_pattern import CL100K_PATTERN
from hale_llm.tokenizer.special_tokens import EOS_TOKEN
from hale_llm.tokenizer.training import build_tokenizer_trainer

CORPUS = [
    "Once upon a time, there was a little girl named Lily. She loved to play outside and run in the park.",
    "One day, a boy named Tom found a red ball. \"Look!\" he said. \"It's mine now.\" He didn't want to share.",
    "The cat sat on the mat.\n\nThe dog barked at 3 birds, 12 trees and 1,000 leaves... What a day!",
] * 80

AWKWARD = [
    "", "Hello , world . It 's a test ! \" quoted \"", "don't can't I'm we've they'll she'd",
    "line one\nline two\n\nline four\r\nline five\r\n\r\n", "a\tb    c  d\t\te   ", "   padded text   ",
    "12345678901234567890 3.14159 1,000,000 007", "café naïve 中文字符 日本語 Ελληνικά العربية 😀🎉👍🏽 é",
    "null\x00char \x1b[31mred\x1b[0m ​ zero-width ﻿", "def f(x):\n    return x**2\n\nprint(f(3))",
    "a" * 300 + " " + "!" * 50 + "\n" * 20, "Once upon a time, there was a little girl named Lily.",
]


def _train(kind: str, vocab_size: int = 700):
    raw = build_tokenizer_trainer(TokenizerConfig(kind=kind, stage2_max_tokens=100_000)).train(CORPUS, vocab_size, [EOS_TOKEN])
    return TokenizerStore.wrap(raw)


@pytest.fixture(scope="module", params=["bpe", "superbpe"])
def kind_and_tokenizer(request):
    return request.param, _train(request.param)


def _bytes_to_unicode() -> dict[int, str]:
    """GPT-2's byte -> printable character table (the byte-level alphabet)."""
    kept = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    chars, extra = kept[:], 0
    for byte in range(256):
        if byte not in kept:
            kept.append(byte)
            chars.append(256 + extra)
            extra += 1
    return {b: chr(c) for b, c in zip(kept, chars)}


class ReferenceBPE:
    """Byte-level BPE from the definition: bytes, then repeatedly merge the lowest-ranked pair (leftmost first)."""

    def __init__(self, tokenizer, split_words: bool) -> None:
        model = json.loads(tokenizer.backend_tokenizer.to_str())["model"]
        self.vocab = model["vocab"]
        self.rank = {tuple(m.split(" ", 1)) if isinstance(m, str) else tuple(m): i for i, m in enumerate(model["merges"])}
        self.table = _bytes_to_unicode()
        self.pattern = regex.compile(CL100K_PATTERN) if split_words else None

    def encode(self, text: str) -> list[int]:
        pieces = self.pattern.findall(text) if self.pattern else [text]
        ids: list[int] = []
        for piece in pieces:
            symbols = [self.table[b] for b in piece.encode("utf-8")]
            while len(symbols) > 1:
                rank, i = min((self.rank.get((symbols[j], symbols[j + 1]), 1 << 60), j) for j in range(len(symbols) - 1))
                if rank == 1 << 60:
                    break
                symbols[i : i + 2] = [symbols[i] + symbols[i + 1]]
            ids += [self.vocab[s] for s in symbols]
        return ids


def test_every_awkward_text_survives_a_round_trip_exactly(kind_and_tokenizer):
    _, tokenizer = kind_and_tokenizer
    for text in AWKWARD:
        ids = tokenizer.encode(text, add_special_tokens=False)
        assert tokenizer.decode(ids, skip_special_tokens=False) == text, text


def test_encoding_matches_an_independent_byte_level_bpe(kind_and_tokenizer):
    kind, tokenizer = kind_and_tokenizer
    reference = ReferenceBPE(tokenizer, split_words=(kind == "bpe"))  # SuperBPE has no word splitting after stage 1
    for text in [*CORPUS[:3], *AWKWARD]:
        assert tokenizer.encode(text, add_special_tokens=False) == reference.encode(text), (kind, text)


def test_the_vocabulary_is_consistent_and_never_exceeds_the_request(kind_and_tokenizer):
    """A small corpus can run out of pairs that occur at least twice, so the size may fall short of the request
    (the dry run reports this); it must never exceed it, and it must always equal merges + bytes + end-of-text."""
    _, tokenizer = kind_and_tokenizer
    size = len(tokenizer)
    assert size <= 700 and sorted(tokenizer.get_vocab().values()) == list(range(size))
    model = json.loads(tokenizer.backend_tokenizer.to_str())["model"]
    assert len(model["merges"]) + 256 + 1 == size


def test_every_byte_has_a_token_so_nothing_is_ever_unknown(kind_and_tokenizer):
    _, tokenizer = kind_and_tokenizer
    singles = {token for token in tokenizer.get_vocab() if len(token) == 1}
    assert len(singles) == 256 and tokenizer.unk_token is None
    assert tokenizer.decode(tokenizer.encode(bytes(range(256)).decode("latin-1"))) == bytes(range(256)).decode("latin-1")


def test_the_only_special_token_is_end_of_text_and_ordinary_text_never_produces_it(kind_and_tokenizer):
    _, tokenizer = kind_and_tokenizer
    assert tokenizer.eos_token == EOS_TOKEN and tokenizer.eos_token_id == 0
    assert tokenizer.bos_token is None and tokenizer.pad_token is None  # no start token, no padding in pretraining
    for text in [*CORPUS[:3], *AWKWARD]:
        assert 0 not in tokenizer.encode(text, add_special_tokens=False)
    assert tokenizer.encode("hi", add_special_tokens=True) == tokenizer.encode("hi", add_special_tokens=False)  # adds nothing


def test_the_literal_text_of_the_special_token_becomes_the_special_token(kind_and_tokenizer):
    """Known, standard behaviour: a document containing the string '<|endoftext|>' would inject an end-of-text id."""
    _, tokenizer = kind_and_tokenizer
    assert 0 in tokenizer.encode("before <|endoftext|> after", add_special_tokens=False)
