"""Logging how long a whole training run took, locally and across spot interruptions."""

from __future__ import annotations

import json
import re
import time

import pytest
import torch
from loguru import logger

from hale_llm.cli.cloud import training_times
from hale_llm.config.schema import load_config
from hale_llm.runtime import CheckpointStore
from hale_llm.training.duration import format_duration
from hale_llm.training.lean_trainer import train_lean
from hale_llm.training.run_timer import RunTimer
from hale_llm.training.resume_checkpoint import ResumeCheckpoint


# ----- formatting -----
@pytest.mark.parametrize(
    "seconds,text",
    [(0, "0.0 s"), (12.34, "12.3 s"), (59.94, "59.9 s"), (60, "1 min 00 s"), (247, "4 min 07 s"),
     (3599, "59 min 59 s"), (3600, "1 h 00 min 00 s"), (3725.4, "1 h 02 min 05 s"), (90061, "25 h 01 min 01 s")],
)
def test_durations_read_naturally(seconds, text):
    assert format_duration(seconds) == text


def test_a_negative_duration_is_an_error():
    with pytest.raises(ValueError):
        format_duration(-1)


# ----- the timer -----
def test_the_timer_adds_the_earlier_sessions_to_this_one():
    timer = RunTimer(previous_seconds=100.0)
    time.sleep(0.05)
    assert 100.04 < timer.total_seconds() < 101.0
    assert 0.04 < timer.session_seconds() < 1.0
    assert timer.total_seconds() >= 100.0 + timer.session_seconds() - 0.01


def test_a_fresh_timer_starts_near_zero_and_never_goes_backwards():
    timer = RunTimer()
    readings = [timer.total_seconds() for _ in range(5)]
    assert readings[0] < 0.5 and readings == sorted(readings)


def test_a_negative_start_is_refused():
    with pytest.raises(ValueError):
        RunTimer(-1.0)


# ----- the files -----
def test_the_completion_record_holds_the_time_and_old_records_still_read(tmp_path):
    resume = ResumeCheckpoint(tmp_path)
    resume.mark_complete(40, 1234.56)
    assert json.loads(resume.complete_path.read_text()) == {"step": 40, "elapsed_seconds": 1234.6}
    assert resume.completed_step() == 40
    resume.mark_complete(41)  # no time given: the old format
    assert json.loads(resume.complete_path.read_text()) == {"step": 41} and resume.completed_step() == 41


def test_the_cloud_cli_finds_the_times_in_a_downloaded_folder_and_skips_bad_files(tmp_path):
    for name, content in [("run_a", '{"step": 5, "elapsed_seconds": 3725.4}'), ("run_b", '{"step": 5}'),
                          ("run_c", "not json"), ("nested/run_d", '{"step": 9, "elapsed_seconds": 61}')]:
        (tmp_path / name).mkdir(parents=True)
        (tmp_path / name / "training_complete.json").write_text(content)
    assert training_times(tmp_path) == [("nested/run_d", 61.0), ("run_a", 3725.4)]
    assert training_times(tmp_path / "nowhere") == []


# ----- the trainer -----
def _cfg(tmp_path, steps=24, save_every=10):
    cfg = load_config("configs/autoregressive/nano.yaml")
    cfg.train.steps, cfg.train.epochs = steps, None
    cfg.train.batch_size, cfg.train.resume, cfg.train.log_every = 2, True, 100
    cfg.train.checkpoint_path = str(tmp_path / "run" / "model.pt")
    cfg.train.lean.save_every, cfg.train.lean.eval_every, cfg.train.lean.keep_best = save_every, 0, 0
    cfg.train.lean.warmup_steps, cfg.train.lean.compile, cfg.train.lean.precision = 2, False, "fp32"
    cfg.model.d_model, cfg.model.n_heads, cfg.model.n_layers, cfg.model.d_ff = 32, 4, 1, 64
    cfg.data.source, cfg.data.overfit_text, cfg.data.n_overfit_copies = "overfit", "the cat sat on the mat", 16
    cfg.set_context_window(16)
    return cfg


def _capture():
    lines: list[str] = []
    handler = logger.add(lambda message: lines.append(str(message)), level="INFO", format="{message}")
    return lines, handler


class _Stop(Exception):
    pass


def _interrupt_after(step_count):
    def hook(step):
        if step + 1 == step_count:
            raise _Stop()
    return hook


def _finished_line(lines):
    found = [line for line in lines if "training finished:" in line]
    assert len(found) == 1, found
    return found[0]


def test_a_run_logs_one_summary_line_with_steps_tokens_and_time(tiny_tokenizer, tmp_path):
    lines, handler = _capture()
    try:
        train_lean(_cfg(tmp_path, steps=12, save_every=0), tokenizer=tiny_tokenizer)
    finally:
        logger.remove(handler)
    line = _finished_line(lines)
    assert re.search(r"training finished: 12 steps, [\d,]+ tokens in .+ \(\d+\.\d s\)$", line), line
    assert "this session" not in line  # a run that never stopped has only one session
    recorded = json.loads((tmp_path / "run" / "training_complete.json").read_text())
    assert recorded["step"] == 12 and recorded["elapsed_seconds"] > 0


def test_after_an_interruption_the_total_includes_the_earlier_session(tiny_tokenizer, tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    with pytest.raises(_Stop):
        train_lean(cfg, tokenizer=tiny_tokenizer, on_step=_interrupt_after(14))  # resume.pt written at step 10
    saved = ResumeCheckpoint(tmp_path / "run").load()["elapsed_seconds"]
    assert saved > 0
    # Pretend the first session had already run for two hours when it saved.
    resume = ResumeCheckpoint(tmp_path / "run")
    state = resume.load()
    state["elapsed_seconds"] = 7200.0
    CheckpointStore().save(str(resume.path), **state)
    lines, handler = _capture()
    try:
        train_lean(cfg, tokenizer=tiny_tokenizer)
    finally:
        logger.remove(handler)
    line = _finished_line(lines)
    assert "2 h 00 min" in line and "this session" in line, line  # 7200 s earlier + a few seconds now
    assert any("2 h 00 min 00 s of training so far" in text for text in lines)
    recorded = json.loads((tmp_path / "run" / "training_complete.json").read_text())["elapsed_seconds"]
    assert 7200.0 < recorded < 7300.0


def test_a_checkpoint_from_before_the_time_was_recorded_still_resumes(tiny_tokenizer, tmp_path):
    cfg = _cfg(tmp_path)
    with pytest.raises(_Stop):
        train_lean(cfg, tokenizer=tiny_tokenizer, on_step=_interrupt_after(14))
    resume = ResumeCheckpoint(tmp_path / "run")
    state = resume.load()
    del state["elapsed_seconds"]  # the old file format
    CheckpointStore().save(str(resume.path), **state)
    assert len(train_lean(cfg, tokenizer=tiny_tokenizer)) == 14  # steps 10..23 ran again
    assert json.loads((tmp_path / "run" / "training_complete.json").read_text())["elapsed_seconds"] > 0


def test_the_time_in_the_resume_checkpoint_grows_between_saves(tiny_tokenizer, tmp_path):
    cfg = _cfg(tmp_path, steps=40, save_every=10)
    seen = []

    def hook(step):
        if step + 1 in (11, 21, 31):  # one step after each save (on_step runs before the save of its own step)
            seen.append(ResumeCheckpoint(tmp_path / "run").load()["elapsed_seconds"])

    train_lean(cfg, tokenizer=tiny_tokenizer, on_step=hook)
    assert len(seen) == 3 and seen == sorted(seen) and seen[0] > 0
