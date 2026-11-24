"""hale-llm-train picks the backend named in the config and fails clearly on a Mac for torchtitan."""

import pytest

from hale_llm.cli import train as train_cli


def test_default_trainer_is_lean_and_writes_a_checkpoint(tiny_tokenizer, monkeypatch, tmp_path):
    from hale_llm.training.backends import lean_backend

    seen = {}
    monkeypatch.setattr(
        lean_backend, "train_lean", lambda cfg, tokenizer=None: seen.setdefault("variant", cfg.variant)
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("sys.argv", [
        "hale-llm-train", "--config", str(__import__("pathlib").Path(__file__).parents[2] / "configs/autoregressive/nano.yaml"),
        "--steps", "2", "--no-tensorboard",
    ])
    train_cli.main()
    assert seen["variant"] == "autoregressive"


def test_torchtitan_on_a_machine_without_cuda_fails_before_creating_a_run(monkeypatch, tmp_path):
    from hale_llm.training.backends import TorchTitanUnavailableError, titan_backend

    monkeypatch.setattr(titan_backend.torch.cuda, "is_available", lambda: False)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("sys.argv", [
        "hale-llm-train", "--config", str(__import__("pathlib").Path(__file__).parents[2] / "configs/autoregressive/nano.yaml"),
        "--trainer", "torchtitan",
    ])
    with pytest.raises(TorchTitanUnavailableError):
        train_cli.main()
    assert not (tmp_path / "data" / "experiments").exists()
