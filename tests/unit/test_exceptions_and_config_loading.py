"""hale_llm exception hierarchy and YAML loading through HaleBlocks' ConfigLoader."""

from __future__ import annotations

from pathlib import Path

import pytest
from hale_core.exceptions import ConfigError, HaleCoreError

from hale_llm.config import load_config
from hale_llm.config.experiment import _copy_source_yamls
from hale_llm.exceptions import (
    DataPipelineError,
    ExperimentNotFoundError,
    GenerationError,
    HaleLLMError,
    ModelConfigurationError,
    NonFiniteLossError,
    TrainingError,
    VisualizationError,
)
from hale_llm.rl import OnlineRLConfig, OnlineRLError

REPO = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    ("cls", "builtin"),
    [
        (ModelConfigurationError, ValueError),
        (DataPipelineError, ValueError),
        (GenerationError, ValueError),
        (VisualizationError, ValueError),
        (ExperimentNotFoundError, FileNotFoundError),
        (TrainingError, RuntimeError),
        (NonFiniteLossError, FloatingPointError),
        (OnlineRLError, RuntimeError),
    ],
)
def test_every_error_is_a_hale_error_and_keeps_its_builtin(cls, builtin):
    assert issubclass(cls, HaleLLMError) and issubclass(cls, HaleCoreError)
    assert issubclass(cls, builtin)


def test_non_finite_loss_carries_the_step():
    err = NonFiniteLossError(17)
    assert err.step == 17 and "step 17" in str(err) and isinstance(err, TrainingError)


def test_model_validation_raises_model_configuration_error():
    from hale_llm.config import RunConfig
    from hale_llm.models.models_mtp.model import MultiTokenPredictionLM

    cfg = RunConfig(variant="mtp", model={"attn_type": "bidirectional"})
    with pytest.raises(ModelConfigurationError, match="causal"):
        MultiTokenPredictionLM(vocab_size=32, cfg=cfg)


def _yaml(path: Path, text: str) -> Path:
    path.write_text(text)
    return path


def test_run_config_inherits_base_lists_and_deep_chains(tmp_path):
    _yaml(tmp_path / "a.yaml", "variant: autoregressive\nmodel: {d_model: 32, n_layers: 1}\n")
    _yaml(tmp_path / "b.yaml", "inherits: a.yaml\nmodel: {n_layers: 2}\n")
    _yaml(tmp_path / "c.yaml", "inherits: b.yaml\ntrain: {lr: 0.5}\n")
    _yaml(tmp_path / "d.yaml", "_base_: [c.yaml]\nmodel: {d_model: 64}\n")
    cfg = load_config(tmp_path / "d.yaml")
    assert (cfg.model.d_model, cfg.model.n_layers, cfg.train.lr) == (64, 2, 0.5)


def test_cycles_and_missing_files(tmp_path):
    _yaml(tmp_path / "x.yaml", "inherits: y.yaml\n")
    _yaml(tmp_path / "y.yaml", "inherits: x.yaml\n")
    with pytest.raises(ConfigError, match="circular"):
        load_config(tmp_path / "x.yaml")
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "missing.yaml")
    with pytest.raises(OnlineRLError, match="cycle"):
        OnlineRLConfig.from_yaml(tmp_path / "x.yaml")


def test_copy_source_yamls_follows_every_base(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    _yaml(src / "a.yaml", "variant: autoregressive\n")
    _yaml(src / "b.yaml", "model: {d_model: 32}\n")
    _yaml(src / "c.yaml", "inherits: a.yaml\n_base_: [b.yaml]\n")
    _copy_source_yamls(src / "c.yaml", tmp_path / "out")
    assert sorted(p.name for p in (tmp_path / "out").iterdir()) == ["a.yaml", "b.yaml", "c.yaml"]


def test_shipped_rl_configs_still_resolve():
    cfg = OnlineRLConfig.from_yaml(REPO / "configs/rl/arithmetic_diffusion.yaml")
    assert cfg.variant == "diffusion" and cfg.sft.steps == 400 and cfg.rollout.group_size == 8
