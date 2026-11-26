"""Resuming an interrupted run: the data order, the leaderboard, atomic saves, and the trainer itself.

The key property: a run that is interrupted and resumed ends up *identical* to one that never stopped.
"""

from __future__ import annotations

import pytest
import torch
from torch.utils.data import DataLoader

from hale_llm.config.schema import load_config
from hale_llm.config.sections.context_stage import ContextStageConfig
from hale_llm.runtime import CheckpointStore
from hale_llm.training.lean_trainer import train_lean
from hale_llm.training.resumable_batches import ResumableBatches
from hale_llm.training.resume_checkpoint import ResumeCheckpoint
from hale_llm.training.top_k_checkpoints import TopKCheckpoints


# ----- the data order -----
def _loader(n=10, batch_size=2, calls=None):
    def collate(items):
        if calls is not None:
            calls.append(len(items))
        return {"ids": torch.stack(items)}

    return DataLoader(list(torch.arange(n).reshape(n, 1)), batch_size=batch_size, shuffle=True,
                      drop_last=True, collate_fn=collate)


def _take(stream, n):
    return [tuple(next(stream)["ids"].flatten().tolist()) for _ in range(n)]


def test_the_batch_order_depends_only_on_seed_and_epoch():
    a = _take(ResumableBatches(_loader(), seed=3), 12)
    b = _take(ResumableBatches(_loader(), seed=3), 12)
    c = _take(ResumableBatches(_loader(), seed=4), 12)
    assert a == b and a != c


def test_each_epoch_uses_every_window_once_in_a_new_order():
    epochs = _take(ResumableBatches(_loader(), seed=0), 10)  # 5 batches per epoch
    first, second = sorted(sum(epochs[:5], ())), sorted(sum(epochs[5:], ()))
    assert first == second == list(range(10))
    assert epochs[:5] != epochs[5:]


@pytest.mark.parametrize("start", [0, 3, 5, 7, 12])  # inside an epoch, on a boundary, across epochs
def test_starting_mid_run_gives_the_rest_of_the_same_stream(start):
    full = _take(ResumableBatches(_loader(), seed=1), 20)
    resumed = ResumableBatches(_loader(), seed=1, start_batch=start)
    assert _take(resumed, 20 - start) == full[start:]
    assert resumed.consumed == 20


def test_skipping_does_not_load_the_skipped_data():
    calls: list[int] = []
    stream = ResumableBatches(_loader(calls=calls), seed=1, start_batch=4)  # 4 of 5 batches already used
    next(stream)
    assert len(calls) == 1  # only the one batch that was asked for was collated


# ----- the best-checkpoint leaderboard -----
def test_leaderboard_is_rebuilt_from_the_folder_and_future_files_are_dropped(tmp_path):
    first = TopKCheckpoints(tmp_path, keep=3, score_name="loss", higher_is_better=False)
    for step, loss in [(10, 5.0), (20, 4.0), (30, 3.0), (40, 2.0)]:
        first.offer(loss, step, model_state_dict={})
    # the run was interrupted and its resume checkpoint is from step 20
    again = TopKCheckpoints(tmp_path, keep=3, score_name="loss", higher_is_better=False)
    again.restore_from_disk(max_step=20)
    assert [p.name for p in again.paths] == ["step0000020_loss_4.0000.pt"] and not (tmp_path / "step0000030_loss_3.0000.pt").exists()
    # and it keeps ranking correctly from there
    assert again.offer(4.5, 25, model_state_dict={}) is not None
    assert [p.name for p in again.paths] == ["step0000020_loss_4.0000.pt", "step0000025_loss_4.5000.pt"]


# ----- atomic saves -----
def test_a_failed_save_leaves_the_previous_checkpoint_intact(tmp_path, monkeypatch):
    store, path = CheckpointStore(), str(tmp_path / "x.pt")
    store.save(path, step=1)

    def explode(obj, f):
        open(f, "wb").write(b"half a fi")  # a truncated file, as when the instance is paused mid-write
        raise OSError("interrupted")

    monkeypatch.setattr(torch, "save", explode)
    with pytest.raises(OSError):
        store.save(path, step=2)
    monkeypatch.undo()
    assert store.load(path)["step"] == 1


def test_resume_checkpoint_refuses_another_model(tmp_path):
    resume = ResumeCheckpoint(tmp_path)
    state = {"variant": "autoregressive", "vocab_size": 100}
    with pytest.raises(ValueError, match="--no-resume"):
        resume.apply(state, model=None, optimizer=None, precision=None, variant="autoregressive_moe", vocab_size=100)
    with pytest.raises(ValueError, match="--no-resume"):
        resume.apply(state, model=None, optimizer=None, precision=None, variant="autoregressive", vocab_size=200)


# ----- the trainer -----
class DistinctWindowsData:
    """Stands in for DataModule: 60 distinct random windows, shuffled by a real DataLoader.

    (The built-in overfit mode is a single window, which would hide any mistake in the data order.)
    """

    def __init__(self, cfg, tokenizer):
        self.cfg, self.tokenizer = cfg, tokenizer

    def _loader(self, shuffle):
        from hale_llm.dataloader.collate import get_collate

        length = self.cfg.model.max_length
        windows = torch.randint(0, len(self.tokenizer), (60, length), generator=torch.Generator().manual_seed(123))
        collate = get_collate(self.cfg.variant)
        return DataLoader(list(windows), batch_size=self.cfg.train.batch_size, shuffle=shuffle, drop_last=True,
                          collate_fn=lambda items: collate(torch.stack(items), tokenizer=self.tokenizer))

    def train_loader(self):
        return self._loader(shuffle=True)

    def val_loader(self):
        return self._loader(shuffle=False)


@pytest.fixture(autouse=True)
def distinct_windows(monkeypatch):
    monkeypatch.setattr("hale_llm.training.lean_trainer.DataModule", DistinctWindowsData)


class Interrupted(Exception):
    """Stands in for the instance being paused."""


def _cfg(tmp_path, *, variant="autoregressive", optimizer="adamw", precision="fp32", steps=40, save_every=10):
    cfg = load_config("configs/autoregressive/small.yaml")
    cfg.variant = variant
    cfg.model.ffn_type = "moe" if variant == "autoregressive_moe" else "mlp"
    cfg.model.moe_hidden_dim = 64
    cfg.train.steps, cfg.train.epochs, cfg.train.batch_size = steps, None, 4
    cfg.train.lr, cfg.train.optimizer_type, cfg.train.log_every = 3e-3, optimizer, 10
    cfg.train.lean.accum_steps, cfg.train.lean.warmup_steps = 2, 5
    cfg.train.lean.eval_every, cfg.train.lean.eval_batches = 15, 2
    cfg.train.lean.save_every, cfg.train.lean.keep_best, cfg.train.lean.precision = save_every, 3, precision
    cfg.train.checkpoint_path = str(tmp_path / "ckpt" / "last.pt")
    cfg.model.d_model, cfg.model.n_heads, cfg.model.n_layers, cfg.model.d_ff = 32, 4, 2, 64
    cfg.set_context_window(16)
    cfg.device = "cpu"
    cfg.logging.tensorboard, cfg.logging.log_file, cfg.logging.level = False, None, "WARNING"
    cfg.experiment.enabled = False
    cfg.train.resume = False
    return cfg


def _weights(cfg):
    return torch.load(cfg.train.checkpoint_path, weights_only=False)["model_state_dict"]


def _interrupt_after(step_index):
    def on_step(step):
        if step == step_index:
            raise Interrupted
    return on_step


@pytest.mark.parametrize(
    "variant,optimizer,precision",
    [
        ("autoregressive", "adamw", "fp32"),
        ("autoregressive", "muon", "fp16"),        # optimizer state of both groups + the loss scaler
        ("autoregressive_moe", "adamw", "fp32"),   # the router noise depends on the restored random state
        ("autoregressive_mtp", "adamw", "fp32"),   # the MTP modules are part of the model and the optimizer
    ],
)
def test_an_interrupted_run_resumes_to_exactly_the_uninterrupted_result(variant, optimizer, precision, tiny_tokenizer, tmp_path):
    # The reference: 40 steps without stopping.
    ref = _cfg(tmp_path / "ref", variant=variant, optimizer=optimizer, precision=precision)
    ref_losses = train_lean(ref, tokenizer=tiny_tokenizer)

    # The same run, "paused" during step 25 (the last resume checkpoint is from step 20), then restarted.
    run = _cfg(tmp_path / "run", variant=variant, optimizer=optimizer, precision=precision)
    with pytest.raises(Interrupted):
        train_lean(run, tokenizer=tiny_tokenizer, on_step=_interrupt_after(24))
    assert (tmp_path / "run" / "ckpt" / "resume.pt").exists()
    run.train.resume = True
    resumed_losses = train_lean(run, tokenizer=tiny_tokenizer)

    assert len(resumed_losses) == 20  # only steps 21..40 were redone
    assert resumed_losses == pytest.approx(ref_losses[20:], abs=1e-6)
    for name, tensor in _weights(ref).items():
        assert torch.allclose(tensor, _weights(run)[name], atol=1e-6), name


@pytest.mark.parametrize("interrupt_at", [24, 34])  # resume point exactly on the stage boundary, and inside the stage
def test_resuming_across_a_context_stage_matches_the_uninterrupted_run(interrupt_at, tiny_tokenizer, tmp_path):
    def staged(root):
        cfg = _cfg(root)
        cfg.train.context_stages = [ContextStageConfig(name="longer", start_fraction=0.5, context_window=32,
                                                       rope_theta=20000.0, long_doc_fraction=0.0)]
        return cfg

    ref = staged(tmp_path / "ref")
    ref_losses = train_lean(ref, tokenizer=tiny_tokenizer)
    run = staged(tmp_path / "run")
    with pytest.raises(Interrupted):
        train_lean(run, tokenizer=tiny_tokenizer, on_step=_interrupt_after(interrupt_at))
    restarted = staged(tmp_path / "run")  # a restart builds a fresh config from the YAML (the first one was changed by its stage switch)
    restarted.train.resume = True
    resumed_losses = train_lean(restarted, tokenizer=tiny_tokenizer)

    saved_at = (interrupt_at + 1) // 10 * 10
    assert resumed_losses == pytest.approx(ref_losses[saved_at:], abs=1e-6)
    for name, tensor in _weights(ref).items():
        assert torch.allclose(tensor, _weights(run)[name], atol=1e-6), name


def test_the_best_checkpoints_stay_consistent_across_an_interruption(tiny_tokenizer, tmp_path):
    ref = _cfg(tmp_path / "ref")
    train_lean(ref, tokenizer=tiny_tokenizer)
    run = _cfg(tmp_path / "run")
    with pytest.raises(Interrupted):
        train_lean(run, tokenizer=tiny_tokenizer, on_step=_interrupt_after(24))  # validation ran at step 15
    run.train.resume = True
    train_lean(run, tokenizer=tiny_tokenizer)
    best = lambda root: sorted(p.name for p in (root / "ckpt" / "best").glob("*.pt"))
    assert best(tmp_path / "run") == best(tmp_path / "ref")


def test_a_finished_run_is_not_redone_on_resume(tiny_tokenizer, tmp_path):
    cfg = _cfg(tmp_path)
    train_lean(cfg, tokenizer=tiny_tokenizer)
    before = (tmp_path / "ckpt" / "last.pt").stat().st_mtime_ns
    assert not (tmp_path / "ckpt" / "resume.pt").exists()  # the rolling file is dropped at the end
    cfg.train.resume = True
    assert train_lean(cfg, tokenizer=tiny_tokenizer) == []  # skipped
    assert (tmp_path / "ckpt" / "last.pt").stat().st_mtime_ns == before


def test_resume_without_a_checkpoint_starts_from_scratch(tiny_tokenizer, tmp_path):
    cfg = _cfg(tmp_path, steps=12)
    cfg.train.resume = True
    assert len(train_lean(cfg, tokenizer=tiny_tokenizer)) == 12


def test_a_fresh_run_discards_a_stale_resume_checkpoint(tiny_tokenizer, tmp_path):
    cfg = _cfg(tmp_path)
    with pytest.raises(Interrupted):
        train_lean(cfg, tokenizer=tiny_tokenizer, on_step=_interrupt_after(24))
    assert (tmp_path / "ckpt" / "resume.pt").exists()
    cfg.train.steps = 5  # a different, short run in the same folder with resume off
    train_lean(cfg, tokenizer=tiny_tokenizer)
    assert not (tmp_path / "ckpt" / "resume.pt").exists()


def test_save_every_zero_writes_no_resume_checkpoint(tiny_tokenizer, tmp_path):
    cfg = _cfg(tmp_path, save_every=0)
    with pytest.raises(Interrupted):
        train_lean(cfg, tokenizer=tiny_tokenizer, on_step=_interrupt_after(24))
    assert not (tmp_path / "ckpt" / "resume.pt").exists()


def test_resume_refuses_a_checkpoint_from_another_model(tiny_tokenizer, tmp_path):
    cfg = _cfg(tmp_path)
    with pytest.raises(Interrupted):
        train_lean(cfg, tokenizer=tiny_tokenizer, on_step=_interrupt_after(24))
    other = _cfg(tmp_path, variant="autoregressive_moe")
    other.train.resume = True
    with pytest.raises(ValueError, match="--no-resume"):
        train_lean(other, tokenizer=tiny_tokenizer)
