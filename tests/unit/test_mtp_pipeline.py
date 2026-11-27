"""The MTP variant inside the real trainer, checkpoints, inference session and dry run."""

from __future__ import annotations

import pytest

from hale_llm.config.schema import load_config
from hale_llm.inference import load_session
from hale_llm.registry import get_variant
from hale_llm.training.lean_trainer import train_lean


def _cfg(tmp_path, depth=2, steps=20, ffn_type="mlp"):
    cfg = load_config("configs/autoregressive_mtp/dense_nano.yaml")
    cfg.train.steps, cfg.train.epochs, cfg.train.batch_size = steps, None, 8
    cfg.train.lr, cfg.train.log_every = 3e-3, 5
    cfg.train.lean.accum_steps, cfg.train.lean.warmup_steps = 2, 3
    cfg.train.lean.eval_every, cfg.train.lean.eval_batches, cfg.train.lean.save_every = 10, 2, 0
    cfg.train.checkpoint_path = str(tmp_path / "ckpt" / "last.pt")
    cfg.model.d_model, cfg.model.n_heads, cfg.model.n_layers, cfg.model.d_ff = 32, 4, 2, 64
    cfg.model.ffn_type, cfg.model.moe_hidden_dim, cfg.model.mtp_depth = ffn_type, 48, depth
    cfg.set_context_window(32)
    cfg.data.overfit_text, cfg.data.n_overfit_copies = "hello world hello world", 32
    cfg.device = "cpu"
    cfg.logging.tensorboard, cfg.logging.log_file, cfg.logging.level = False, None, "WARNING"
    cfg.experiment.enabled = False
    cfg.train.resume = False
    return cfg


@pytest.mark.parametrize("ffn_type", ["mlp", "moe"])
def test_training_logs_every_piece_of_the_loss(ffn_type, tiny_tokenizer, tmp_path):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    cfg = _cfg(tmp_path, ffn_type=ffn_type)
    cfg.logging.tensorboard, cfg.logging.tensorboard_dir = True, str(tmp_path / "tb")
    losses = train_lean(cfg, tokenizer=tiny_tokenizer)
    acc = EventAccumulator(str(tmp_path / "tb"), size_guidance={"scalars": 0, "tensors": 0})
    acc.Reload()
    tags = set(acc.Tags()["scalars"])
    assert {"train/ntp_loss", "train/mtp_loss", "train/mtp_loss_depth1", "train/mtp_loss_depth2", "train/ce_loss", "val/loss"} <= tags
    # ce_loss is the next-token loss alone, not the weighted total that is optimised.
    ntp = [e.value for e in acc.Scalars("train/ntp_loss")]
    assert ntp == pytest.approx([e.value for e in acc.Scalars("train/ce_loss")])
    total = [e.value for e in acc.Scalars("train/loss")]
    mtp = [e.value for e in acc.Scalars("train/mtp_loss")]
    weights = [e.value for e in acc.Scalars("train/mtp_weight")]
    assert {round(w, 4) for w in weights} == {0.3, 0.1}  # lambda: 0.3, then 0.1
    for t, n, m, w in zip(total, ntp, mtp, weights):
        assert t >= n + w * m - 1e-3  # total = ntp + lambda * mtp (+ the MoE balancing term when MoE)
    assert losses[-1] < losses[0]


def test_a_checkpoint_loads_and_generates_the_same_with_and_without_speculation(tiny_tokenizer, tmp_path):
    cfg = _cfg(tmp_path)
    train_lean(cfg, tokenizer=tiny_tokenizer)
    session = load_session(cfg, tokenizer=tiny_tokenizer, require_checkpoint=True)
    assert type(session.model).variant_name == "autoregressive_mtp" and session.model.mtp.depth == 2
    session.cfg.sample.speculative = True
    speculative = session.generate("hello", max_new_tokens=12, temperature=0.0)
    session.cfg.sample.speculative = False
    plain = session.generate("hello", max_new_tokens=12, temperature=0.0)
    assert speculative.text == plain.text and speculative.token_ids.tolist() == plain.token_ids.tolist()


def test_the_checkpoint_remembers_the_mtp_settings_even_if_the_yaml_changes(tiny_tokenizer, tmp_path):
    trained = _cfg(tmp_path, depth=2)
    train_lean(trained, tokenizer=tiny_tokenizer)
    other = _cfg(tmp_path, depth=1)  # a YAML that says depth 1
    other.model.mtp_loss_weight = 0.9
    session = load_session(other, tokenizer=tiny_tokenizer, require_checkpoint=True)
    assert session.model.mtp.depth == 2 and session.cfg.model.mtp_loss_weight == 0.3  # the trained architecture wins


def test_the_dry_run_expects_ln_vocab_times_one_plus_lambda():
    import math

    from hale_llm.training.dry_run import DryRunReport

    cfg = load_config("configs/autoregressive_mtp/dense_nano.yaml")
    scale = get_variant(cfg.variant).initial_loss_scale(cfg)
    assert scale == pytest.approx(1.0 + cfg.model.mtp_loss_weight)
    assert get_variant("autoregressive").initial_loss_scale(cfg) == 1.0
    report = DryRunReport("autoregressive_mtp", "cpu", 1000, 1, 1, [9.0], 1.0, None, [], loss_scale=scale)
    assert report.expected_first_loss == pytest.approx(math.log(1000) * scale)
    assert "x 1.3" in report.format()


@pytest.mark.parametrize("optimizer", ["adamw", "muon"])
@pytest.mark.parametrize("precision", ["fp32", "bf16", "fp16"])
def test_mtp_trains_with_every_optimizer_and_precision(optimizer, precision, tiny_tokenizer, tmp_path):
    cfg = _cfg(tmp_path, depth=4)
    cfg.train.optimizer_type, cfg.train.lean.precision, cfg.train.steps = optimizer, precision, 40
    cfg.set_context_window(32)
    losses = train_lean(cfg, tokenizer=tiny_tokenizer)
    assert all(map(lambda x: x == x and abs(x) < 1e4, losses)) and losses[-1] < losses[0] * 0.85


def test_the_muon_split_puts_the_mtp_matrices_on_muon_and_the_norms_on_adamw(tmp_path):
    import hale_llm.models  # noqa: F401
    from hale_llm.optim.parameter_groups import is_hidden_matrix

    cfg = _cfg(tmp_path, depth=2)
    model = get_variant(cfg.variant)(vocab_size=64, cfg=cfg)
    kinds = {name: is_hidden_matrix(name, p) for name, p in model.named_parameters()}
    assert kinds["mtp.depths.0.proj.weight"] and kinds["mtp.depths.0.blocks.0.attn.proj.weight"]
    assert not kinds["mtp.depths.0.hidden_norm.weight"] and not kinds["mtp.depths.1.head_norm.weight"]  # 1-D: AdamW
    assert not kinds["backbone.token_emb.weight"]  # the shared table stays on AdamW


def test_the_dry_run_counts_active_parameters_in_the_mtp_blocks_too(tmp_path):
    from hale_llm.training.dry_run import DryRun

    cfg = _cfg(tmp_path, depth=2, ffn_type="moe")
    total, active = DryRun(cfg)._count_parameters(vocab_size=64)
    model = get_variant(cfg.variant)(vocab_size=64, cfg=cfg)
    blocks = [*model.backbone.layers, *(b for d in model.mtp.depths for b in d.blocks)]
    routed = sum(p.numel() for b in blocks for p in b.ffn.routed_experts.parameters())
    assert total == sum(p.numel() for p in model.parameters())
    assert active == int(total - routed * (1 - cfg.model.moe_top_k / cfg.model.moe_num_experts))
