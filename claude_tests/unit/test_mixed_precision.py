"""Mixed precision: policy, loss scaling and overflow handling (CPU supports bf16 and fp16 autocast)."""

from __future__ import annotations

import pytest
import torch

from hale_llm.training.mixed_precision import MixedPrecision


def test_cpu_auto_is_fp32():
    assert MixedPrecision.from_setting("auto", "cpu").name == "fp32"


@pytest.mark.parametrize("setting", ["fp32", "bf16", "fp16"])
def test_explicit_setting_is_honoured(setting):
    assert MixedPrecision.from_setting(setting, "cpu").name == setting


def test_only_fp16_uses_loss_scaling():
    assert MixedPrecision.from_setting("fp16", "cpu").scaler.is_enabled()
    assert not MixedPrecision.from_setting("bf16", "cpu").scaler.is_enabled()
    assert not MixedPrecision.from_setting("fp32", "cpu").scaler.is_enabled()


def test_autocast_changes_the_matmul_dtype():
    x = torch.randn(4, 4)
    for setting, dtype in [("fp32", torch.float32), ("bf16", torch.bfloat16), ("fp16", torch.float16)]:
        with MixedPrecision.from_setting(setting, "cpu").autocast():
            assert (x @ x).dtype == dtype


def _tiny_problem():
    weight = torch.nn.Parameter(torch.ones(4))
    return weight, torch.optim.SGD([weight], lr=0.1)


@pytest.mark.parametrize("setting", ["fp32", "bf16", "fp16"])
def test_step_reports_the_true_grad_norm_and_updates(setting):
    mixed = MixedPrecision.from_setting(setting, "cpu")
    weight, optimizer = _tiny_problem()
    mixed.backward((weight * 3).sum())  # gradient is 3 per element, norm 6
    norm = mixed.clip_and_step(optimizer, [weight], max_norm=100.0)
    assert norm == pytest.approx(6.0)  # fp16: scaled gradients were unscaled before measuring
    assert torch.allclose(weight, torch.full((4,), 0.7))  # 1 - 0.1 * 3
    assert not mixed.step_skipped


def test_clipping_works_in_real_units_with_fp16():
    mixed = MixedPrecision.from_setting("fp16", "cpu")
    weight, optimizer = _tiny_problem()
    mixed.backward((weight * 3).sum())
    mixed.clip_and_step(optimizer, [weight], max_norm=1.0)  # norm 6 is clipped to 1 -> gradient 0.5 each
    assert torch.allclose(weight, torch.full((4,), 0.95))


def test_fp16_overflow_skips_the_update_and_lowers_the_scale():
    mixed = MixedPrecision.from_setting("fp16", "cpu")
    weight, optimizer = _tiny_problem()
    scale_before = mixed.loss_scale
    mixed.backward((weight * float("inf")).sum())
    mixed.clip_and_step(optimizer, [weight], max_norm=1.0)
    assert mixed.step_skipped and mixed.loss_scale < scale_before
    assert torch.equal(weight.detach(), torch.ones(4))  # weights untouched
    # The next healthy step goes through.
    optimizer.zero_grad()
    mixed.backward((weight * 3).sum())
    mixed.clip_and_step(optimizer, [weight], max_norm=100.0)
    assert not mixed.step_skipped and not torch.equal(weight.detach(), torch.ones(4))


def test_bf16_on_an_unsupported_gpu_is_a_clear_error(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: False)
    with pytest.raises(ValueError, match="fp16"):
        MixedPrecision.from_setting("bf16", "cuda")
    assert MixedPrecision.from_setting("auto", "cuda").name == "fp32"


def test_the_loss_scale_survives_a_save_and_load():
    first = MixedPrecision.from_setting("fp16", "cpu")
    weight, optimizer = _tiny_problem()
    first.backward((weight * float("inf")).sum())
    first.clip_and_step(optimizer, [weight], max_norm=1.0)  # overflow: the scale is halved
    lowered = first.loss_scale
    assert lowered < 65536.0
    second = MixedPrecision.from_setting("fp16", "cpu")
    second.load_state_dict(first.state_dict())
    assert second.loss_scale == lowered


def test_loading_a_scaler_state_into_a_non_fp16_run_is_ignored():
    fp16_state = MixedPrecision.from_setting("fp16", "cpu").state_dict()
    MixedPrecision.from_setting("bf16", "cpu").load_state_dict(fp16_state)  # must not raise
    MixedPrecision.from_setting("fp16", "cpu").load_state_dict({})  # a checkpoint saved without scaling
