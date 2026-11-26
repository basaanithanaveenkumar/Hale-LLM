"""Schedule maths of the lean trainer, and a short end-to-end run that writes a loadable checkpoint."""

import pytest

from hale_llm.training.lean_trainer import lr_multiplier, train_lean
from hale_llm.config.schema import load_config
from hale_llm.inference import load_session


def test_lr_multiplier_shape():
    f = lambda s: lr_multiplier(s, total_steps=100, warmup_steps=10, warmdown_frac=0.2)  # noqa: E731
    assert f(0) == pytest.approx(0.1)
    assert f(4) == pytest.approx(0.5)
    assert f(9) == pytest.approx(1.0)
    assert f(50) == 1.0
    assert f(90) == pytest.approx(0.5)
    assert f(99) == pytest.approx(0.05)


def test_no_warmdown_stays_flat():
    assert lr_multiplier(99, 100, warmup_steps=0, warmdown_frac=0.0) == 1.0


def _cfg(tmp_path, optimizer):
    cfg = load_config("configs/autoregressive/small.yaml")
    cfg.train.steps, cfg.train.epochs, cfg.train.batch_size = 60, None, 8
    cfg.train.lr, cfg.train.optimizer_type, cfg.train.log_every = 3e-3, optimizer, 20
    cfg.train.lean.accum_steps, cfg.train.lean.warmup_steps = 2, 5
    cfg.train.lean.eval_every, cfg.train.lean.eval_batches = 30, 2
    cfg.train.checkpoint_path = str(tmp_path / "ckpt.pt")
    cfg.model.d_model, cfg.model.n_heads, cfg.model.n_layers = 64, 4, 2
    cfg.model.d_ff = 128
    cfg.set_context_window(32)
    cfg.data.overfit_text, cfg.data.n_overfit_copies = "hello world hello world", 32
    cfg.device = "cpu"
    cfg.logging.tensorboard, cfg.logging.log_file, cfg.logging.level = False, None, "WARNING"
    cfg.experiment.enabled = False
    return cfg


@pytest.mark.parametrize("optimizer", ["adamw", "muon"])
def test_train_lean_learns_and_saves_loadable_checkpoint(optimizer, tiny_tokenizer, tmp_path):
    cfg = _cfg(tmp_path, optimizer)
    losses = train_lean(cfg, tokenizer=tiny_tokenizer)
    assert len(losses) == 60
    assert losses[-1] < losses[0] * 0.7
    # The checkpoint must load through the same path chat/sample use.
    session = load_session(cfg, tokenizer=tiny_tokenizer, require_checkpoint=True)
    assert isinstance(session.generate("hello").text, str)


@pytest.mark.parametrize("variant", ["autoregressive", "autoregressive_moe"])
def test_both_variants_overfit_and_reload(variant, tiny_tokenizer, tmp_path):
    cfg = _cfg(tmp_path, "adamw")
    cfg.variant = variant
    cfg.model.ffn_type = "moe" if variant == "autoregressive_moe" else "mlp"
    cfg.model.moe_hidden_dim = 64
    losses = train_lean(cfg, tokenizer=tiny_tokenizer)
    assert losses[-1] < losses[0] * 0.7
    session = load_session(cfg, tokenizer=tiny_tokenizer, require_checkpoint=True)
    assert type(session.model).variant_name == variant
    assert isinstance(session.generate("hello").text, str)


def test_epochs_are_counted_in_optimizer_steps_not_batches():
    from hale_llm.config.schema import RunConfig
    from hale_llm.training.schedule import TrainSchedule

    # 20 batches per epoch and 4 batches per step: 2 epochs = 10 steps (not 40).
    cfg = RunConfig(variant="autoregressive", train={"epochs": 2, "steps": None})
    assert TrainSchedule.from_config(cfg, 20 // 4).total_steps == 10


def test_a_checkpoint_loads_even_if_the_yaml_describes_another_architecture(tiny_tokenizer, tmp_path):
    trained = _cfg(tmp_path, "adamw")
    trained.variant, trained.model.ffn_type = "autoregressive_moe", "moe"
    trained.model.moe_num_experts, trained.model.moe_top_k, trained.model.norm_type = 6, 3, "rms"
    trained.train.steps = 3
    train_lean(trained, tokenizer=tiny_tokenizer)

    wrong_yaml = _cfg(tmp_path, "adamw")  # dense, layer norm, different width
    wrong_yaml.variant, wrong_yaml.model.ffn_type, wrong_yaml.model.d_model = "autoregressive_moe", "moe", 32
    session = load_session(wrong_yaml, tokenizer=tiny_tokenizer, require_checkpoint=True)
    assert session.cfg.model.d_model == 64 and session.cfg.model.norm_type == "rms"
    assert len(session.model.backbone.stack.layers[0].ffn.routed_experts) == 6


def test_a_checkpoint_of_another_variant_is_rejected(tiny_tokenizer, tmp_path):
    trained = _cfg(tmp_path, "adamw")
    trained.train.steps = 2
    train_lean(trained, tokenizer=tiny_tokenizer)
    other = _cfg(tmp_path, "adamw")
    other.variant, other.model.ffn_type = "autoregressive_moe", "moe"
    with pytest.raises(ValueError, match="variant"):
        load_session(other, tokenizer=tiny_tokenizer, require_checkpoint=True)


def _read_tensorboard(directory):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    acc = EventAccumulator(str(directory), size_guidance={"scalars": 0, "tensors": 0})
    acc.Reload()
    return acc


@pytest.mark.parametrize("variant", ["autoregressive", "autoregressive_moe"])
def test_training_writes_tensorboard_scalars(variant, tiny_tokenizer, tmp_path):
    cfg = _cfg(tmp_path, "adamw")
    cfg.variant = variant
    cfg.model.ffn_type = "moe" if variant == "autoregressive_moe" else "mlp"
    cfg.train.steps, cfg.train.lean.eval_every, cfg.train.log_every = 6, 3, 2
    cfg.logging.tensorboard, cfg.logging.tensorboard_dir = True, str(tmp_path / "tb")
    losses = train_lean(cfg, tokenizer=tiny_tokenizer)

    acc = _read_tensorboard(tmp_path / "tb")
    tags = set(acc.Tags()["scalars"])
    expected = {
        "train/loss", "train/ce_loss", "train/perplexity", "train/grad_norm", "train/lr",
        "train/tokens_seen", "train/tokens_per_sec", "val/loss", "val/perplexity",
        "model/parameters", "model/weight_norm",
    }
    assert expected <= tags, expected - tags
    assert ("moe/expert_share_max" in tags) == (variant == "autoregressive_moe")
    assert ("train/moe_balance_loss" in tags) == (variant == "autoregressive_moe")
    if variant == "autoregressive_moe":  # a per-layer mean sits near 1.0 (even routing), not near n_layers
        assert 0.9 < acc.Scalars("train/moe_balance_loss")[0].value < 2.0

    train_events = acc.Scalars("train/loss")
    assert [e.step for e in train_events] == list(range(1, 7))  # one point per optimizer step
    assert [e.value for e in train_events] == pytest.approx(losses, rel=1e-5)
    assert [e.step for e in acc.Scalars("val/loss")] == [0, 3, 6]  # start, every 3, end
    assert acc.Scalars("train/tokens_seen")[-1].value == 6 * 8 * 2 * 32  # steps x batch x accum x window


def test_tensorboard_can_be_switched_off(tiny_tokenizer, tmp_path):
    cfg = _cfg(tmp_path, "adamw")
    cfg.train.steps = 2
    cfg.logging.tensorboard, cfg.logging.tensorboard_dir = False, str(tmp_path / "tb")
    train_lean(cfg, tokenizer=tiny_tokenizer)
    assert not (tmp_path / "tb").exists()


def test_each_run_keeps_a_copy_of_the_shared_tokenizer(tmp_path):
    """Chat/sample/eval load <run>/tokenizer first, so the checkpoint survives a changed shared tokenizer."""
    import hale_llm.models  # noqa: F401
    from hale_llm.config.experiment import apply_experiment_layout
    from hale_llm.tokenizer.store import TokenizerStore

    cfg = load_config("configs/autoregressive/nano.yaml")
    cfg.experiment.enabled, cfg.experiment.root = True, str(tmp_path / "experiments")
    cfg.tokenizer.path, cfg.tokenizer.vocab_size = str(tmp_path / "shared_tok"), 300
    cfg.tokenizer.kind = "bpe"
    cfg.data.overfit_text, cfg.data.n_overfit_copies = "hello world hello world", 16
    cfg.train.steps, cfg.train.epochs, cfg.train.batch_size = 2, None, 4
    cfg.train.lean.eval_every = 0
    cfg.model.d_model, cfg.model.n_heads, cfg.model.n_layers, cfg.model.d_ff = 32, 4, 1, 64
    cfg.set_context_window(16)
    cfg.device, cfg.logging.tensorboard, cfg.logging.log_file = "cpu", False, None
    run_dir = apply_experiment_layout(cfg, create=True, name="r1")

    train_lean(cfg)
    assert TokenizerStore.exists(tmp_path / "shared_tok")  # trained once, shared
    assert TokenizerStore.exists(run_dir / "tokenizer")  # and copied into the run

    # Replace the shared tokenizer; the run still loads the tokenizer it was trained with.
    import shutil

    shutil.rmtree(tmp_path / "shared_tok")
    session = load_session(cfg, require_checkpoint=True)
    assert session.cfg.tokenizer.path == str(run_dir / "tokenizer")


@pytest.mark.parametrize("best_by", ["loss", "token_accuracy"])
def test_only_the_best_checkpoints_are_kept_and_they_load(best_by, tiny_tokenizer, tmp_path):
    cfg = _cfg(tmp_path, "adamw")
    cfg.train.steps = 40
    cfg.train.lean.eval_every, cfg.train.lean.keep_best, cfg.train.lean.best_by = 10, 2, best_by
    train_lean(cfg, tokenizer=tiny_tokenizer)
    kept = sorted((tmp_path / "best").glob("*.pt"))
    assert len(kept) == 2  # validation ran 4 times, only the top 2 remain
    import torch
    state = torch.load(kept[0], weights_only=False)
    assert best_by in state and "optimizer_state_dict" not in state and state["variant"] == cfg.variant


@pytest.mark.parametrize("optimizer", ["adamw", "muon"])
@pytest.mark.parametrize("precision", ["bf16", "fp16"])
def test_mixed_precision_trains_and_saves_fp32_weights(precision, optimizer, tiny_tokenizer, tmp_path):
    cfg = _cfg(tmp_path, optimizer)
    cfg.train.lean.precision = precision
    losses = train_lean(cfg, tokenizer=tiny_tokenizer)
    assert losses[-1] < losses[0] * 0.8  # still learns
    import torch
    state = torch.load(cfg.train.checkpoint_path, weights_only=False)
    assert all(t.dtype == torch.float32 for t in state["model_state_dict"].values() if t.is_floating_point())
    session = load_session(cfg, tokenizer=tiny_tokenizer, require_checkpoint=True)  # chat/sample/eval can still load it
    assert session is not None


@pytest.mark.parametrize("optimizer", ["adamw", "muon"])
@pytest.mark.parametrize("precision", ["bf16", "fp16"])
def test_mixed_precision_trains_and_saves_fp32_weights(precision, optimizer, tiny_tokenizer, tmp_path):
    cfg = _cfg(tmp_path, optimizer)
    cfg.train.lean.precision = precision
    losses = train_lean(cfg, tokenizer=tiny_tokenizer)
    assert losses[-1] < losses[0] * 0.8  # still learns
    import torch
    state = torch.load(cfg.train.checkpoint_path, weights_only=False)
    assert all(t.dtype == torch.float32 for t in state["model_state_dict"].values() if t.is_floating_point())
    session = load_session(cfg, tokenizer=tiny_tokenizer, require_checkpoint=True)  # chat/sample/eval can still load it
    assert session is not None


@pytest.mark.parametrize("precision", ["bf16", "fp16"])
def test_moe_trains_under_mixed_precision(precision, tiny_tokenizer, tmp_path):
    cfg = _cfg(tmp_path, "adamw")
    cfg.variant, cfg.model.ffn_type, cfg.model.moe_hidden_dim = "autoregressive_moe", "moe", 64
    cfg.train.lean.precision = precision
    losses = train_lean(cfg, tokenizer=tiny_tokenizer)
    assert losses[-1] < losses[0] * 0.8
