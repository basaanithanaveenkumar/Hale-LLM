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


def test_a_moe_checkpoint_saved_with_ffn_type_mlp_still_loads(tiny_tokenizer, tmp_path):
    """Older runs stored ffn_type='mlp' for the MoE variant; the variant decides."""
    from hale_llm.runtime import CheckpointStore

    cfg = _cfg(tmp_path, "adamw")
    cfg.variant, cfg.model.ffn_type, cfg.train.steps = "autoregressive_moe", "moe", 2
    train_lean(cfg, tokenizer=tiny_tokenizer)
    store = CheckpointStore()
    ckpt = store.load(cfg.train.checkpoint_path)
    ckpt["config"]["model"]["ffn_type"] = "mlp"
    store.save(cfg.train.checkpoint_path, **ckpt)
    assert load_session(cfg, tokenizer=tiny_tokenizer, require_checkpoint=True).model.uses_moe
