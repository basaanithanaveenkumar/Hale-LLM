"""data.context_window is the one place the model's context length is chosen."""

import pytest

from hale_llm.config.schema import RunConfig, load_config


def test_context_window_sets_model_length():
    cfg = RunConfig(variant="autoregressive", data={"context_window": 256})
    assert cfg.model.max_length == 256


def test_base_config_uses_128():
    assert load_config("configs/autoregressive/small.yaml").model.max_length == 128


def test_context_window_wins_over_model_length():
    cfg = RunConfig(variant="autoregressive", data={"context_window": 256}, model={"max_length": 128})
    assert cfg.model.max_length == 256


def test_model_length_is_used_when_no_context_window():
    assert RunConfig(variant="autoregressive", model={"max_length": 32}).model.max_length == 32


def test_config_survives_a_dump_and_reload():
    # Ray sends the config to workers as a dict; the round trip must not change the length.
    cfg = RunConfig(variant="autoregressive", data={"context_window": 48})
    assert RunConfig.model_validate(cfg.model_dump()).model.max_length == 48


def test_stride_option_no_longer_exists():
    with pytest.raises(ValueError):
        RunConfig(variant="autoregressive", data={"stride_words": 10})
