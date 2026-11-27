"""Whole pipeline with a tokenizer trained on the data: train, extend the context, reload, generate."""

import random

import pytest
import torch

import hale_llm.models  # noqa: F401
from hale_llm.config.schema import RunConfig
from hale_llm.runtime import CheckpointStore
from hale_llm.dataloader.sources import huggingface as hf_source
from hale_llm.inference import load_session
from hale_llm.training.lean_trainer import train_lean


def _fake_split(*_args, **_kwargs):
    rng = random.Random(3)
    words = ["".join(rng.choice("abcdefghij") for _ in range(rng.randint(2, 6))) for _ in range(60)]
    stories = [" ".join(rng.choice(words) for _ in range(rng.randint(20, 40))) for _ in range(300)]
    stories += [" ".join(rng.choice(words) for _ in range(500)) for _ in range(6)]  # a few long documents
    return [{"text": s} for s in stories]


@pytest.fixture
def cfg(tmp_path, monkeypatch):
    monkeypatch.setattr(hf_source, "load_hf_split", _fake_split)
    return RunConfig(
        variant="autoregressive",
        device="cpu",
        data={"dataset": "fake", "context_window": 32},
        tokenizer={"path": str(tmp_path / "tok"), "vocab_size": 400},
        model={"d_model": 32, "n_heads": 4, "n_layers": 2, "d_ff": 64, "rope_theta": 10_000.0},
        train={
            "steps": 40,
            "epochs": None,
            "batch_size": 4,
            "lr": 3e-3,
            "log_every": 20,
            "checkpoint_path": str(tmp_path / "run" / "checkpoints" / "last.pt"),
            "lean": {"warmup_steps": 3, "eval_every": 20, "eval_batches": 2},
            "context_stages": [
                {"name": "extend", "start_fraction": 0.5, "context_window": 64, "rope_theta": 50_000.0, "long_doc_fraction": 0.3}
            ],
        },
        logging={"tensorboard": False, "log_file": None, "level": "WARNING"},
        experiment={"enabled": False},
    )


def test_train_extend_context_reload_and_generate(cfg, tmp_path):
    losses = train_lean(cfg)
    assert len(losses) == 40 and losses[-1] < losses[0]

    # The run trained its own tokenizer on the data (and saved it for later steps).
    assert (tmp_path / "tok" / "tokenizer.json").is_file()

    # The checkpoint records the final stage: window 64, RoPE base 50k.
    saved = CheckpointStore().load(cfg.train.checkpoint_path, "cpu")
    assert saved["config"]["model"]["max_length"] == 64
    assert saved["config"]["model"]["rope_theta"] == 50_000.0
    assert saved["vocab_size"] <= 400

    # A fresh config (original 32 / 10k) adopts the trained values and the trained tokenizer.
    fresh = RunConfig(
        variant="autoregressive",
        device="cpu",
        data={"dataset": "fake", "context_window": 32},
        tokenizer={"path": str(tmp_path / "tok")},
        model=cfg.model.model_dump() | {"rope_theta": 10_000.0, "max_length": 32},
        train={"checkpoint_path": cfg.train.checkpoint_path},
        logging=cfg.logging.model_dump(),
    )
    session = load_session(fresh, cfg.train.checkpoint_path)
    assert fresh.model.max_length == 64
    assert session.model.backbone.max_length == 64
    assert session.model.backbone.positional.rope_theta == 50_000.0
    result = session.generate("ab cd", max_new_tokens=8)
    assert isinstance(result.text, str) and result.token_ids.size(1) > 0


def test_inference_refuses_to_train_a_tokenizer(cfg):
    with pytest.raises(FileNotFoundError, match="no trained tokenizer"):
        load_session(cfg, require_checkpoint=False)


def test_windows_come_from_the_trained_tokenizer_with_eos_between_documents(cfg):
    from hale_llm.dataloader.data_module import DataModule

    data = DataModule(cfg)
    batch = next(iter(data.train_loader()))
    ids = batch["input_ids"]
    assert ids.shape[1] == 32
    assert (ids == data.tokenizer.eos_token_id).any()  # stop tokens are real, learnable tokens
    assert int(batch["attention_mask"].min()) == 1  # no padding anywhere
    assert int(ids.max()) < len(data.tokenizer)


def test_yarn_lets_the_extended_model_run_beyond_its_trained_window(cfg):
    from hale_llm.nn.backbones import TransformerBackbone

    model_cfg = cfg.model.model_copy(update={"rope_scaling": "yarn", "rope_scaling_factor": 4.0, "rope_original_length": 4096, "max_length": 128})
    backbone = TransformerBackbone(100, model_cfg)
    assert "yarn" in backbone.positional.describe()
    out = backbone(torch.randint(0, 100, (1, 128)))
    assert out.shape == (1, 128, 100) and torch.isfinite(out).all()
