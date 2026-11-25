"""The dry run: report maths and a real end-to-end pass on tiny shapes."""

import math

import pytest

from hale_llm.config.schema import load_config
from hale_llm.training.dry_run import DryRun, DryRunReport


def _report(losses, vocab=1000):
    return DryRunReport("autoregressive", "cpu", vocab, 10_000_000, 4_000_000, losses, 0.5, None, [])


def test_report_mentions_every_number_a_user_needs():
    text = _report([6.9, 6.8, 6.7]).format()
    assert "10.0M total, 4.0M active" in text and "0.50 s/step" in text
    assert f"{math.log(1000):.2f}" in text and "peak GPU memory" not in text


def test_steady_step_time_ignores_the_first_step():
    assert DryRun._steady_step_time([10.0, 11.0, 13.0]) == pytest.approx(1.5)


def test_non_finite_loss_fails_the_dry_run():
    with pytest.raises(FloatingPointError):
        DryRun._check(_report([6.9, float("nan"), 6.8]))


def test_dense_and_moe_count_active_parameters(tmp_path):
    for path, dense in (("configs/autoregressive/nano.yaml", True), ("configs/autoregressive_moe/nano.yaml", False)):
        cfg = load_config(path)
        total, active = DryRun(cfg)._count_parameters(1000)
        assert (total == active) == dense
        assert 0 < active <= total


def test_shrink_keeps_the_real_shapes_but_writes_only_to_the_temp_folder(tmp_path):
    cfg = load_config("configs/autoregressive/small.yaml")
    notes: list[str] = []
    small = DryRun(cfg)._shrink(cfg, tmp_path, notes)
    assert small.train.steps == 3 and small.train.lean.eval_every == 0
    assert small.train.batch_size == cfg.train.batch_size and small.model.max_length == cfg.model.max_length
    assert small.tokenizer.path.startswith(str(tmp_path)) and small.train.checkpoint_path.startswith(str(tmp_path))
    assert not small.experiment.enabled
    assert cfg.train.steps is None or cfg.train.steps != 3  # the original config is untouched


def test_context_stages_are_skipped_with_a_note(tmp_path):
    cfg = load_config("configs/long_context/300m_4k_to_32k.yaml")
    notes: list[str] = []
    small = DryRun(cfg)._shrink(cfg, tmp_path, notes)
    assert small.train.context_stages == [] and cfg.train.context_stages and notes
