"""A short, throw-away training run that proves the whole pipeline works before a long one.

``hale-llm-train --config X.yaml --dry-run`` goes through the same code as a real run, on a
small slice of the data, and writes nothing into ``data/``:

    1. train a tokenizer on a few thousand documents (into a temporary folder),
    2. pack context windows, build the model, the optimizer and the loss,
    3. run a few optimizer steps with the real batch size and context window,
    4. report what a full run would need: parameters at the real vocabulary size,
       the first loss against ``ln(vocab)``, speed and peak memory.

Everything is deleted afterwards.
"""

from __future__ import annotations

import math
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from loguru import logger

from hale_llm.config.schema import RunConfig
from hale_llm.dataloader.tokenizer import build_tokenizer_for
from hale_llm.registry import get_variant
from hale_llm.tokenizer import TokenizerBuilder
from hale_llm.training.lean_trainer import train_lean
from hale_llm.utils.device import get_device

DRY_RUN_STEPS = 3
DRY_RUN_DOCUMENTS = 2_000  # more when the context window is long, so enough windows exist


@dataclass(frozen=True)
class DryRunReport:
    """What the dry run measured; :meth:`format` renders it for the terminal."""

    variant: str
    device: str
    vocab_size: int
    total_params: int
    active_params: int
    losses: list[float]
    seconds_per_step: float
    peak_memory_gb: float | None
    notes: list[str]
    loss_scale: float = 1.0  # a fresh model's loss is ln(vocab) x this (above 1 when the loss has extra terms)

    @property
    def expected_first_loss(self) -> float:
        """A model that guesses uniformly scores ``ln(vocab)``; a fresh model should be close."""
        return math.log(self.vocab_size) * self.loss_scale

    def format(self) -> str:
        lines = [
            "=" * 60,
            f"dry run OK  variant={self.variant}  device={self.device}",
            f"parameters at the real vocabulary ({self.vocab_size:,} tokens): "
            f"{self.total_params / 1e6:.1f}M total, {self.active_params / 1e6:.1f}M active per token",
            f"losses over {len(self.losses)} steps: " + ", ".join(f"{x:.3f}" for x in self.losses),
            f"first loss {self.losses[0]:.2f} vs expected {self.expected_first_loss:.2f} (ln(vocab) x {self.loss_scale:g}) "
            "(a fresh model should be within about 1.0 of it)",
            f"speed: {self.seconds_per_step:.2f} s/step (steps 2-{len(self.losses)}, first step excluded)",
        ]
        if self.peak_memory_gb is not None:
            lines.append(f"peak GPU memory: {self.peak_memory_gb:.1f} GB")
        lines += [f"note: {note}" for note in self.notes]
        lines.append("=" * 60)
        return "\n".join(lines)


class DryRun:
    """Runs a config through the lean trainer on a small slice and reports the result."""

    def __init__(self, cfg: RunConfig) -> None:
        self._cfg = cfg

    def run(self) -> DryRunReport:
        notes: list[str] = []
        with tempfile.TemporaryDirectory(prefix="hale_dry_run_") as tmp:
            cfg = self._shrink(self._cfg, Path(tmp), notes)
            device = get_device(cfg.device)
            if device == "cuda":
                torch.cuda.reset_peak_memory_stats()

            step_ends: list[float] = []
            losses = train_lean(cfg, on_step=lambda _: step_ends.append(time.perf_counter()))
            seconds_per_step = self._steady_step_time(step_ends)
            peak = torch.cuda.max_memory_allocated() / 1e9 if device == "cuda" else None
            tokenizer_vocab = len(build_tokenizer_for(cfg, allow_training=False))

        real_vocab = self._real_vocab_size()
        if tokenizer_vocab < real_vocab:
            notes.append(
                f"the dry-run tokenizer learned {tokenizer_vocab:,} tokens from a small slice; "
                f"a full run learns {real_vocab:,}"
            )
        total, active = self._count_parameters(real_vocab)
        report = DryRunReport(
            variant=cfg.variant, device=device, vocab_size=tokenizer_vocab, total_params=total,
            active_params=active, losses=losses, seconds_per_step=seconds_per_step,
            peak_memory_gb=peak, notes=notes, loss_scale=get_variant(cfg.variant).initial_loss_scale(cfg),
        )
        self._check(report)
        return report

    @staticmethod
    def _steady_step_time(step_ends: list[float]) -> float:
        """Mean seconds per step, ignoring the first step (kernel compilation and warm-up)."""
        gaps = [b - a for a, b in zip(step_ends, step_ends[1:])]
        return sum(gaps) / len(gaps)

    def _shrink(self, cfg: RunConfig, tmp: Path, notes: list[str]) -> RunConfig:
        """A copy of ``cfg`` that trains a few steps on a little data, writing only into ``tmp``."""
        small = cfg.model_copy(deep=True)
        small.experiment.enabled = False
        small.tokenizer.path = str(tmp / "tokenizer")
        small.train.checkpoint_path = str(tmp / "last.pt")
        small.logging.log_file = None
        small.logging.tensorboard = False
        small.train.steps, small.train.epochs = DRY_RUN_STEPS, None
        small.train.lean.eval_every = 0
        small.train.resume, small.train.lean.save_every = False, 0  # a dry run never resumes or leaves resume files
        if small.train.context_stages:
            small.train.context_stages = []
            notes.append("train.context_stages skipped; only the first stage was exercised")
        long_window = (small.data.context_window or small.model.max_length) >= 1024
        documents = DRY_RUN_DOCUMENTS * (10 if long_window else 1)
        small.data.train_size = min(small.data.train_size or documents, documents)
        small.data.val_size = min(small.data.val_size or 200, 200)
        return small

    def _real_vocab_size(self) -> int:
        return TokenizerBuilder(self._cfg).resolve_vocab_size()

    def _count_parameters(self, vocab_size: int) -> tuple[int, int]:
        """(total, active per token) parameters at ``vocab_size``, built without allocating memory."""
        with torch.device("meta"):
            model = get_variant(self._cfg.variant)(vocab_size=vocab_size, cfg=self._cfg)
        total = sum(p.numel() for p in model.parameters())
        model_cfg = self._cfg.model
        if model_cfg.ffn_type != "moe":
            return total, total
        # Every block with a mixture of experts: the trunk's, and the MTP depths' when the variant has them.
        mtp_depths = getattr(getattr(model, "mtp", None), "depths", [])
        blocks = [*model.backbone.stack.layers, *(block for depth in mtp_depths for block in depth.blocks)]
        routed = sum(p.numel() for block in blocks for p in block.ffn.routed_experts.parameters())
        idle_share = 1 - model_cfg.moe_top_k / model_cfg.moe_num_experts
        return total, int(total - routed * idle_share)

    @staticmethod
    def _check(report: DryRunReport) -> None:
        if not all(math.isfinite(x) for x in report.losses):
            raise FloatingPointError(f"non-finite loss in the dry run: {report.losses}")
        if abs(report.losses[0] - report.expected_first_loss) > 1.5:
            logger.warning(
                "first loss {:.2f} is far from ln(vocab)={:.2f}: check initialisation and the tokenizer",
                report.losses[0], report.expected_first_loss,
            )
