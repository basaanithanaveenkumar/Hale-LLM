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
    tok = TokenizerBuilder(cfg).build(allow_training=True)
    assert (tmp_path / "tok" / "tokenizer.json").is_file()
    assert (tmp_path / "tok" / "hale_tokenizer.json").is_file()
    assert tok.eos_token_id == 0 and len(tok) <= 300


def test_second_build_loads_instead_of_training(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    first = TokenizerBuilder(cfg).build(allow_training=True)

    def fail(*a, **k):
        raise AssertionError("must not train again")

    monkeypatch.setattr(superbpe_module.SuperBpeTokenizerTrainer, "train", fail)
    second = TokenizerBuilder(cfg).build(allow_training=False)
    assert second(TEXT[:40])["input_ids"] == first(TEXT[:40])["input_ids"]


def test_inference_never_trains(tmp_path):
    with pytest.raises(FileNotFoundError, match="no trained tokenizer"):
        TokenizerBuilder(_cfg(tmp_path)).build(allow_training=False)


def test_vocab_size_comes_from_the_model_size_when_not_given(tmp_path):
    cfg = _cfg(tmp_path, vocab_size=None)
    assert TokenizerBuilder(cfg).resolve_vocab_size() == 32_768


def test_explicit_vocab_size_wins(tmp_path):
    assert TokenizerBuilder(_cfg(tmp_path, vocab_size=4096)).resolve_vocab_size() == 4096


def test_pipeline_entry_point_trains_for_a_new_run_folder(tmp_path):
    cfg = _cfg(tmp_path)
    tok = build_tokenizer_for(cfg)
    assert tok is build_tokenizer_for(cfg)  # memoised within the process
    assert (tmp_path / "tok" / "tokenizer.json").is_file()


def test_a_shared_tokenizer_is_trained_once_and_reused_by_other_runs(tmp_path, monkeypatch):
    first = TokenizerBuilder(_cfg(tmp_path)).build(allow_training=True)
    monkeypatch.setattr(
        superbpe_module.SuperBpeTokenizerTrainer, "train",
        lambda *a, **k: pytest.fail("a second run must reuse the shared tokenizer"),
    )
    second_cfg = _cfg(tmp_path)  # another run, same shared path
    second_cfg.variant = "autoregressive_moe"
    second = TokenizerBuilder(second_cfg).build(allow_training=True)
    assert first.get_vocab() == second.get_vocab()


def test_reusing_a_tokenizer_of_another_kind_is_an_error_not_a_silent_swap(tmp_path):
    TokenizerBuilder(_cfg(tmp_path, kind="superbpe")).build(allow_training=True)
    with pytest.raises(ValueError, match="kind"):
        TokenizerBuilder(_cfg(tmp_path, kind="bpe")).build(allow_training=True)


def test_reusing_a_tokenizer_with_another_explicit_vocab_size_is_an_error(tmp_path):
    TokenizerBuilder(_cfg(tmp_path, vocab_size=300)).build(allow_training=True)
    with pytest.raises(ValueError, match="requested_vocab_size"):
        TokenizerBuilder(_cfg(tmp_path, vocab_size=400)).build(allow_training=True)
    # No explicit size in the config: whatever the shared tokenizer has is accepted.
    TokenizerBuilder(_cfg(tmp_path, vocab_size=None)).build(allow_training=True)


def test_reusing_a_tokenizer_trained_on_another_dataset_is_an_error(tmp_path):
    TokenizerBuilder(_cfg(tmp_path)).build(allow_training=True)
    other = _cfg(tmp_path)
    other.data.dataset = "some/other-dataset"
    with pytest.raises(ValueError, match="dataset"):
        TokenizerBuilder(other).build(allow_training=True)
