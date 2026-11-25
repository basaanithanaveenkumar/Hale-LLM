"""Byte-level BPE and SuperBPE trainers, the vocabulary policy, and the tokenizer store."""

import pytest

from hale_llm.config.sections.model import ModelConfig
from hale_llm.config.sections.tokenizer import TokenizerConfig
from hale_llm.tokenizer import TokenizerStore, bytes_per_token
from hale_llm.tokenizer.pretokenizer_pattern import CL100K_PATTERN
from hale_llm.tokenizer.special_tokens import EOS_TOKEN, MASK_TOKEN
from hale_llm.tokenizer.training import TOKENIZER_TRAINERS, build_tokenizer_trainer
from hale_llm.tokenizer.training.byte_level_pipeline import regex_byte_level_pre_tokenizer
from hale_llm.tokenizer.vocab import (
    VOCAB_POLICIES,
    estimate_non_embedding_params,
    get_vocab_policy,
)

CORPUS = [
    "Once upon a time there was a little girl. She went to the park with her mom. "
    "She loved to play in the park and her mom loved to watch her play.",
    "One day, a little boy went to the park with his dad. He loved to play at the park. "
    "At the end of the day they went home and ate dinner.",
] * 150


def _train(kind, vocab_size=400, **overrides):
    settings = TokenizerConfig(kind=kind, **overrides)
    raw = build_tokenizer_trainer(settings).train(CORPUS, vocab_size, [EOS_TOKEN])
    return TokenizerStore.wrap(raw, with_mask=False)


def test_both_kinds_are_registered():
    assert {"bpe", "superbpe"} <= set(TOKENIZER_TRAINERS)
    assert TokenizerConfig().kind == "superbpe"  # the default
    assert TokenizerConfig().source == "trained"


def test_unknown_kind_lists_options():
    with pytest.raises(KeyError, match="registered"):
        build_tokenizer_trainer(TokenizerConfig(kind="wordpiece"))


def test_cl100k_pretokenizer_splits_like_gpt4():
    pieces = [p for p, _ in regex_byte_level_pre_tokenizer().pre_tokenize_str("Hello, world! 12345 don't")]
    # Letters, then comma; space attaches to the next word; digits in groups of three; contraction.
    assert pieces == ["Hello", ",", "Ġworld", "!", "Ġ", "123", "45", "Ġdon", "'t"]
    assert "p{L}" in CL100K_PATTERN


@pytest.mark.parametrize("kind", ["bpe", "superbpe"])
def test_roundtrip_is_lossless_for_any_text(kind):
    tok = _train(kind)
    for text in ["Once upon a time", "naïve café — 日本語 😀", "line one\nline two\n\n  spaces", "", "12345 6789"]:
        ids = tok(text, add_special_tokens=False)["input_ids"]
        assert tok.decode(ids) == text


@pytest.mark.parametrize("kind", ["bpe", "superbpe"])
def test_eos_is_the_first_token_and_vocab_is_bounded(kind):
    tok = _train(kind)
    assert tok.eos_token == EOS_TOKEN and tok.eos_token_id == 0
    assert tok.pad_token_id is None
    assert len(tok) <= 400


def test_superbpe_learns_multi_word_tokens_and_compresses_better():
    bpe, superbpe = _train("bpe"), _train("superbpe")
    superwords = [t for t in superbpe.get_vocab() if "Ġ" in t[1:]]
    assert superwords, "stage 2 should create tokens that span spaces"
    assert not [t for t in bpe.get_vocab() if "Ġ" in t[1:]]
    assert bytes_per_token(superbpe, CORPUS[:2]) > bytes_per_token(bpe, CORPUS[:2])


def test_superbpe_reaches_the_requested_vocabulary_when_text_allows():
    import random

    rng = random.Random(0)
    words = ["".join(rng.choice("abcdefghij") for _ in range(rng.randint(2, 6))) for _ in range(80)]
    corpus = [" ".join(rng.choice(words) for _ in range(30)) for _ in range(400)]
    raw = build_tokenizer_trainer(TokenizerConfig(kind="superbpe")).train(corpus, 700, [EOS_TOKEN])
    assert raw.get_vocab_size() == 700


def test_mask_token_is_added_only_when_asked():
    raw = build_tokenizer_trainer(TokenizerConfig(kind="bpe")).train(CORPUS, 300, [EOS_TOKEN, MASK_TOKEN])
    tok = TokenizerStore.wrap(raw, with_mask=True)
    assert tok.mask_token == MASK_TOKEN and tok.mask_token_id == 1


def test_store_round_trip(tmp_path):
    tok = _train("superbpe")
    TokenizerStore.save(tok, tmp_path / "tok", {"kind": "superbpe"})
    assert TokenizerStore.exists(tmp_path / "tok")
    loaded = TokenizerStore.load(tmp_path / "tok")
    text = "the little girl went to the park"
    assert loaded(text)["input_ids"] == tok(text)["input_ids"]
    assert loaded.eos_token_id == tok.eos_token_id


def test_loading_a_missing_tokenizer_fails_clearly(tmp_path):
    with pytest.raises(FileNotFoundError):
        TokenizerStore.load(tmp_path / "nothing")


def test_bytes_per_token_needs_text():
    with pytest.raises(ValueError):
        bytes_per_token(_train("bpe"), [""])


# ----- vocabulary size policy -----

@pytest.mark.parametrize(
    ("params", "expected"),
    [(10_000_000, 32_768), (499_999_999, 32_768), (500_000_000, 65_536), (2_999_999_999, 65_536), (3_000_000_000, 100_352)],
)
def test_vocab_tiers(params, expected):
    assert get_vocab_policy("by_model_size").choose(params) == expected


def test_vocab_policy_rejects_negative_and_unknown():
    with pytest.raises(ValueError):
        get_vocab_policy("by_model_size").choose(-1)
    with pytest.raises(KeyError, match="registered"):
        get_vocab_policy("by_vibes")
    assert "by_model_size" in VOCAB_POLICIES


def test_parameter_estimate_matches_a_hand_count():
    # d=64, 2 layers, mha, mlp d_ff=128: attention 4*64^2 = 16,384; ffn 2*64*128 = 16,384.
    model = ModelConfig(d_model=64, n_heads=4, n_layers=2, d_ff=128, attn_impl="mha", ffn_type="mlp")
    assert estimate_non_embedding_params(model) == 2 * (16_384 + 16_384)
    gated = model.model_copy(update={"ffn_type": "geglu"})
    assert estimate_non_embedding_params(gated) == 2 * (16_384 + 3 * 64 * 128)


def test_300m_recipe_lands_in_the_small_vocab_tier():
    model = ModelConfig(d_model=1024, n_heads=16, n_layers=24, d_ff=4096, attn_impl="gqa", n_kv_heads=4)
    params = estimate_non_embedding_params(model)
    assert 200e6 < params < 350e6
    assert get_vocab_policy("by_model_size").choose(params) == 32_768
