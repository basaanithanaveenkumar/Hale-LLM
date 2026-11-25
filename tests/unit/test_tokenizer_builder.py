"""TokenizerBuilder: train once per run folder, load afterwards, never train for inference."""

import pytest

from hale_llm.config.schema import RunConfig
from hale_llm.dataloader.tokenizer import build_tokenizer_for
from hale_llm.tokenizer import TokenizerBuilder
from hale_llm.tokenizer.training import superbpe_tokenizer_trainer as superbpe_module

TEXT = "the little girl went to the park with her mom and they played all day. " * 6


def _cfg(tmp_path, **tokenizer):
    return RunConfig(
        variant="autoregressive",
        data={"overfit_text": TEXT},
        tokenizer={"path": str(tmp_path / "tok"), "vocab_size": 300, **tokenizer},
        model={"d_model": 32, "n_heads": 4, "n_layers": 1, "d_ff": 64},
    )


def test_trains_on_the_data_and_saves_next_to_the_run(tmp_path):
    cfg = _cfg(tmp_path)
    tok = TokenizerBuilder(cfg, with_mask=False).build(allow_training=True)
    assert (tmp_path / "tok" / "tokenizer.json").is_file()
    assert (tmp_path / "tok" / "hale_tokenizer.json").is_file()
    assert tok.eos_token_id == 0 and len(tok) <= 300


def test_second_build_loads_instead_of_training(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    first = TokenizerBuilder(cfg, with_mask=False).build(allow_training=True)

    def fail(*a, **k):
        raise AssertionError("must not train again")

    monkeypatch.setattr(superbpe_module.SuperBpeTokenizerTrainer, "train", fail)
    second = TokenizerBuilder(cfg, with_mask=False).build(allow_training=False)
    assert second(TEXT[:40])["input_ids"] == first(TEXT[:40])["input_ids"]


def test_inference_never_trains(tmp_path):
    with pytest.raises(FileNotFoundError, match="no trained tokenizer"):
        TokenizerBuilder(_cfg(tmp_path), with_mask=False).build(allow_training=False)


def test_masking_variants_get_a_mask_token(tmp_path):
    tok = TokenizerBuilder(_cfg(tmp_path), with_mask=True).build(allow_training=True)
    assert tok.mask_token_id == 1


def test_reusing_a_tokenizer_without_mask_for_a_masking_variant_is_rejected(tmp_path):
    TokenizerBuilder(_cfg(tmp_path), with_mask=False).build(allow_training=True)
    with pytest.raises(ValueError, match="needs one"):
        TokenizerBuilder(_cfg(tmp_path), with_mask=True).build(allow_training=False)


def test_vocab_size_comes_from_the_model_size_when_not_given(tmp_path):
    cfg = _cfg(tmp_path, vocab_size=None)
    assert TokenizerBuilder(cfg, with_mask=False).resolve_vocab_size() == 32_768


def test_explicit_vocab_size_wins(tmp_path):
    assert TokenizerBuilder(_cfg(tmp_path, vocab_size=4096), with_mask=False).resolve_vocab_size() == 4096


def test_pipeline_entry_point_trains_for_a_new_run_folder(tmp_path):
    cfg = _cfg(tmp_path)
    tok = build_tokenizer_for(cfg)
    assert tok is build_tokenizer_for(cfg)  # memoised within the process
    assert (tmp_path / "tok" / "tokenizer.json").is_file()
