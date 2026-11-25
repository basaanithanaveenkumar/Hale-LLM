"""Shared pieces of the byte-level tokenizers built with Hugging Face ``tokenizers`` (Apache-2.0)."""

from __future__ import annotations

from tokenizers import Regex, decoders, pre_tokenizers

from hale_llm.tokenizer.pretokenizer_pattern import CL100K_PATTERN


def regex_byte_level_pre_tokenizer() -> pre_tokenizers.PreTokenizer:
    """Split with the cl100k regex, then map every byte to a printable character.

    Step 1 (``Split``) cuts the text into pre-tokens such as " the" or "123".
    Step 2 (``ByteLevel``) rewrites each byte as one of 256 distinct characters, so any
    text, in any language, can be represented with no "unknown" token.
    """
    return pre_tokenizers.Sequence(
        [
            pre_tokenizers.Split(Regex(CL100K_PATTERN), behavior="isolated"),
            pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
        ]
    )


def plain_byte_level_pre_tokenizer() -> pre_tokenizers.PreTokenizer:
    """Byte mapping only: the whole text stays one piece (used by SuperBPE after stage 1)."""
    return pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)


def byte_level_decoder() -> decoders.Decoder:
    """Turns the byte characters back into the original text."""
    return decoders.ByteLevel()


def byte_alphabet() -> list[str]:
    """The 256 byte characters; always in the vocabulary so no text is unrepresentable."""
    return pre_tokenizers.ByteLevel.alphabet()
