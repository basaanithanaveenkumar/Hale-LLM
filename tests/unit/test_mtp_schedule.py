"""The lambda schedule: DeepSeek-V3's step from 0.3 to 0.1, driven by training progress."""

from __future__ import annotations

import pytest
import torch

import hale_llm.models  # noqa: F401
from hale_llm.config.schema import load_config
from hale_llm.config.sections.model import ModelConfig
from hale_llm.models.models_mtp.loss_schedule import MTPLossSchedule
from hale_llm.registry import get_loss, get_variant
from hale_llm.training.lean_trainer import train_lean


def test_the_step_schedule_switches_at_the_configured_fraction():
    schedule = MTPLossSchedule(first=0.3, final=0.1, switch_fraction=0.68)
    assert [schedule.weight_at(p) for p in (0.0, 0.3, 0.679, 0.68, 0.9, 0.999)] == [0.3, 0.3, 0.3, 0.1, 0.1, 0.1]


def test_without_a_final_weight_lambda_is_constant():
    schedule = MTPLossSchedule(first=0.3, final=None, switch_fraction=0.68)
    assert {schedule.weight_at(p) for p in (0.0, 0.5, 0.68, 0.99)} == {0.3}


def test_it_is_built_from_the_config_and_matches_deepseek_v3_by_default():
    schedule = MTPLossSchedule.from_config(ModelConfig())
    assert (schedule.first, schedule.final, schedule.switch_fraction) == (0.3, 0.1, 0.68)
    assert abs(10 / 14.8 - 0.68) < 0.01  # 10T of 14.8T tokens


@pytest.mark.parametrize("field,value", [("mtp_switch_fraction", 0.0), ("mtp_switch_fraction", 1.0),
                                         ("mtp_loss_weight_final", -0.1), ("mtp_loss_weight", -0.1)])
def test_invalid_schedule_settings_are_rejected(field, value):
    with pytest.raises(ValueError):
        ModelConfig(**{field: value})


def _cfg(tmp_path, depth=1, steps=20, switch=0.5):
    cfg = load_config("configs/autoregressive_mtp/dense_nano.yaml")
    cfg.train.steps, cfg.train.epochs, cfg.train.batch_size, cfg.train.lr, cfg.train.log_every = steps, None, 8, 3e-3, 5
    cfg.train.lean.accum_steps, cfg.train.lean.warmup_steps = 2, 3
    cfg.train.lean.eval_every, cfg.train.lean.save_every = 0, 0
    cfg.train.checkpoint_path = str(tmp_path / "ckpt" / "last.pt")
    cfg.model.d_model, cfg.model.n_heads, cfg.model.n_layers, cfg.model.d_ff = 32, 4, 2, 64
    cfg.model.mtp_depth, cfg.model.mtp_switch_fraction = depth, switch
    cfg.set_context_window(32)
    cfg.data.overfit_text, cfg.data.n_overfit_copies = "hello world hello world", 32
    cfg.device = "cpu"
    cfg.logging.tensorboard, cfg.logging.tensorboard_dir = True, str(tmp_path / "tb")
    cfg.logging.log_file, cfg.logging.level = None, "WARNING"
    cfg.experiment.enabled = cfg.train.resume = False
    return cfg


def _logged_weights(tmp_path):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    acc = EventAccumulator(str(tmp_path / "tb"), size_guidance={"scalars": 0})
    acc.Reload()
    return [(e.step, round(e.value, 4)) for e in acc.Scalars("train/mtp_weight")]


def test_the_trainer_switches_lambda_at_exactly_the_right_step(tiny_tokenizer, tmp_path):
    train_lean(_cfg(tmp_path, steps=20, switch=0.5), tokenizer=tiny_tokenizer)
    weights = dict(_logged_weights(tmp_path))
    # Step s (0-based) runs at progress s / 20: s = 0..9 are before 0.5, s = 10..19 are after it. Logged steps are 1-based.
    assert [weights[n] for n in range(1, 11)] == [0.3] * 10 and [weights[n] for n in range(11, 21)] == [0.1] * 10


def test_every_depth_follows_the_same_schedule(tiny_tokenizer, tmp_path):
    runs = {}
    for depth in (1, 2, 3):
        root = tmp_path / f"d{depth}"
        train_lean(_cfg(root, depth=depth), tokenizer=tiny_tokenizer)
        runs[depth] = _logged_weights(root)
    assert runs[1] == runs[2] == runs[3]  # the 1/D factor, not the schedule, keeps the depths comparable


def test_the_loss_uses_the_current_weight():
    cfg = load_config("configs/autoregressive_mtp/dense_nano.yaml")
    cfg.set_context_window(16)
    cfg.model.d_model, cfg.model.n_heads, cfg.model.n_layers, cfg.model.d_ff, cfg.model.dropout = 32, 4, 2, 64, 0.0
    torch.manual_seed(0)
    model = get_variant(cfg.variant)(vocab_size=40, cfg=cfg).train()
    ids = torch.randint(0, 40, (2, 16))
    labels = ids.clone()
    labels[:, :-1], labels[:, -1] = ids[:, 1:], -100
    batch = {"input_ids": ids, "attention_mask": torch.ones_like(ids), "labels": labels}
    loss_fn = get_loss(cfg.variant)
    model.set_training_progress(0.0)
    early = loss_fn(model, batch)
    mtp_mean = model.loss_parts["mtp_loss"]
    model.set_training_progress(0.9)
    late = loss_fn(model, batch)
    assert model.mtp_weight == 0.1 and torch.allclose(early - late, (0.3 - 0.1) * mtp_mean, atol=1e-6)


def test_a_fresh_model_expects_the_first_phase_weight():
    cfg = load_config("configs/autoregressive_mtp/dense_nano.yaml")
    assert get_variant(cfg.variant).initial_loss_scale(cfg) == pytest.approx(1.3)  # the loss at step 0 uses lambda = 0.3
