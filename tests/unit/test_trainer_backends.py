"""Backend registry, lean-backend guards and the TorchTitan launcher (no GPU needed)."""

import sys

import pytest

from hale_llm.config.schema import RunConfig
from hale_llm.config.sections.titan import TitanConfig
from hale_llm.training.backends import (
    TRAINER_BACKENDS,
    TorchTitanBackend,
    TorchTitanCommandBuilder,
    TorchTitanUnavailableError,
    get_trainer_backend,
)
from hale_llm.training.backends import titan_backend


def test_three_backends_are_registered_and_lean_is_default():
    assert {"lean", "full", "torchtitan"} <= set(TRAINER_BACKENDS)
    assert RunConfig(variant="autoregressive").train.trainer == "lean"


def test_unknown_backend_lists_options():
    with pytest.raises(KeyError, match="registered"):
        get_trainer_backend("megatron")


def test_lean_rejects_distributed_settings():
    cfg = RunConfig(variant="autoregressive", train={"parallel_strategy": "ddp"})
    with pytest.raises(ValueError, match="trainer=full"):
        get_trainer_backend("lean").validate(cfg)


def test_command_builder_matches_run_train_sh():
    settings = TitanConfig(
        module="torchtitan_recipes.models.llama3",
        config="llama3_8b",
        output_dir="out",
        resume_step=100,
        overrides=["foo=1"],
        extra_args=["--print-config"],
    )
    command = TorchTitanCommandBuilder(settings, num_gpus=4).build()
    assert command[:3] == [sys.executable, "-m", "torch.distributed.run"]
    assert "--nproc_per_node=4" in command
    assert command[command.index("-m", 3) + 1] == "torchtitan.train"
    assert command[command.index("--module") + 1] == "torchtitan_recipes.models.llama3"
    assert command[command.index("--config") + 1] == "llama3_8b"
    assert command[command.index("--output-dir") + 1] == "out"
    assert command[command.index("--resume-step") + 1] == "100"
    assert command[command.index("--override") + 1] == "foo=1"
    assert command[-1] == "--print-config"


def test_command_builder_rejects_zero_gpus():
    with pytest.raises(ValueError):
        TorchTitanCommandBuilder(TitanConfig(), num_gpus=0)


def test_torchtitan_refuses_to_run_without_cuda(monkeypatch):
    # This is what happens on a Mac: no CUDA -> clear error, nothing launched.
    monkeypatch.setattr(titan_backend.torch.cuda, "is_available", lambda: False)
    with pytest.raises(TorchTitanUnavailableError, match="NVIDIA"):
        TorchTitanBackend().run(RunConfig(variant="autoregressive"))


def test_torchtitan_reports_missing_install(monkeypatch):
    monkeypatch.setattr(titan_backend.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(titan_backend.importlib.util, "find_spec", lambda name: None)
    with pytest.raises(TorchTitanUnavailableError, match="not installed"):
        TorchTitanBackend().validate(RunConfig(variant="autoregressive"))


def _pretend_gpu_box(monkeypatch, tmp_path, returncode):
    (tmp_path / "torchtitan_recipes").mkdir()
    monkeypatch.setattr(titan_backend.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(titan_backend.torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(titan_backend.importlib.util, "find_spec", lambda name: object())
    calls = []

    class Done:
        pass

    def fake_run(command, cwd, env, check):
        calls.append((command, cwd, env))
        done = Done()
        done.returncode = returncode
        return done

    monkeypatch.setattr(titan_backend.subprocess, "run", fake_run)
    cfg = RunConfig(variant="autoregressive", train={"titan": {"repo_dir": str(tmp_path)}})
    return cfg, calls


def test_torchtitan_launches_with_all_visible_gpus(monkeypatch, tmp_path):
    cfg, calls = _pretend_gpu_box(monkeypatch, tmp_path, returncode=0)
    assert TorchTitanBackend().run(cfg) == 0
    command, cwd, env = calls[0]
    assert "--nproc_per_node=2" in command
    assert cwd == tmp_path.resolve()
    assert env["PYTORCH_ALLOC_CONF"]


def test_torchtitan_failure_raises(monkeypatch, tmp_path):
    cfg, _ = _pretend_gpu_box(monkeypatch, tmp_path, returncode=3)
    with pytest.raises(RuntimeError, match="exit code 3"):
        TorchTitanBackend().run(cfg)


def test_torchtitan_rejects_more_gpus_than_visible(monkeypatch, tmp_path):
    cfg, _ = _pretend_gpu_box(monkeypatch, tmp_path, returncode=0)
    cfg.train.titan.num_gpus = 8
    with pytest.raises(TorchTitanUnavailableError, match="only 2"):
        TorchTitanBackend().run(cfg)
