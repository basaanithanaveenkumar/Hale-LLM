"""The three precision-study configs must differ only in precision and run name."""

from __future__ import annotations

import pytest

from hale_llm.config.schema import load_config

_NAMES = ["fp32", "bf16", "fp16"]


def _comparable(name: str) -> dict:
    cfg = load_config(f"configs/precision_study/{name}.yaml").model_dump()
    cfg["train"]["lean"].pop("precision")
    cfg["experiment"].pop("name")
    cfg["experiment"].pop("source_yaml", None)
    return cfg


@pytest.mark.parametrize("name", _NAMES)
def test_each_run_sets_its_own_precision_and_targets_cuda(name):
    cfg = load_config(f"configs/precision_study/{name}.yaml")
    assert cfg.train.lean.precision == name and cfg.experiment.name == f"precision_{name}"
    assert cfg.device == "cuda"


def test_runs_are_identical_apart_from_precision():
    assert _comparable("fp32") == _comparable("bf16") == _comparable("fp16")
