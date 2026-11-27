"""The 2B baseline and its MTP twin: the right size, the same everything else."""

from __future__ import annotations

import pytest
import torch

import hale_llm.models  # noqa: F401
from hale_llm.cloud.jarvislabs.settings import JarvisSettings
from hale_llm.config.schema import load_config
from hale_llm.registry import get_variant

VOCAB = 32_768  # the shared tokenizer


def _params(cfg) -> int:
    with torch.device("meta"):  # no memory: only shapes
        model = get_variant(cfg.variant)(vocab_size=VOCAB, cfg=cfg)
    return sum(p.numel() for p in model.parameters())


def test_the_baseline_is_a_two_billion_parameter_model():
    assert 1.9e9 < _params(load_config("configs/mtp_2b/baseline.yaml")) < 2.1e9


def test_the_mtp_model_adds_one_depth_worth_of_parameters_per_configured_depth():
    baseline, mtp = load_config("configs/mtp_2b/baseline.yaml"), load_config("configs/mtp_2b/mtp.yaml")
    extra = _params(mtp) - _params(baseline)
    d, m = baseline.model.d_model, mtp.model
    per_depth = m.mtp_n_layers * (4 * d * d + 2 * d * baseline.model.d_ff) + 2 * d * d + 3 * d  # blocks + proj(2d->d) + norms
    assert m.mtp_depth == 1 and extra == pytest.approx(m.mtp_depth * per_depth, rel=0.01)  # ~59M: 2.04B in total


def test_the_two_runs_differ_only_in_the_mtp_settings_and_their_name():
    def comparable(path):
        dump = load_config(path).model_dump()
        for key in [k for k in dump["model"] if k.startswith("mtp_")]:
            dump["model"].pop(key)
        dump["experiment"].pop("name"), dump["experiment"].pop("source_yaml", None), dump.pop("variant")
        return dump

    assert comparable("configs/mtp_2b/baseline.yaml") == comparable("configs/mtp_2b/mtp.yaml")


def test_each_step_is_sixteen_sequences_and_uses_bf16():
    for name in ("baseline", "mtp"):
        cfg = load_config(f"configs/mtp_2b/{name}.yaml")
        assert cfg.train.batch_size * cfg.train.lean.accum_steps == 16 and cfg.train.lean.precision == "bf16"
        assert cfg.model.max_length == 1024 and cfg.device == "cuda"


def test_the_cloud_settings_for_the_2b_runs_load_and_give_enough_disk():
    settings = JarvisSettings.from_yaml("configs/cloud/jarvislabs_a100_80gb.yaml")
    assert settings.gpu_type == "A100-80GB" and settings.storage_gb >= 200
