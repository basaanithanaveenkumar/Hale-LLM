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
