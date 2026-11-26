"""The rolling checkpoint that lets an interrupted training run continue where it stopped."""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch
from loguru import logger

from hale_llm.runtime import CheckpointStore


class ResumeCheckpoint:
    """Save and restore everything needed to continue a run exactly.

    Files in the checkpoint folder:

        resume.pt              rolling checkpoint, overwritten every ``save_every`` steps
        training_complete.json written when the run finishes, so a restart does not redo it

    ``resume.pt`` holds the weights, optimizer, fp16 loss scaler, random-number state, the step,
    the tokens seen and the data position. It is written through :class:`CheckpointStore`, which
    renames a finished temp file into place, so an interruption mid-save keeps the older file.
    """

    def __init__(self, directory: Path, store: CheckpointStore | None = None) -> None:
        self.path = directory / "resume.pt"
        self.complete_path = directory / "training_complete.json"
        self.store = store or CheckpointStore()

    # ----- saving -----
    def save(self, *, step: int, tokens_seen: int, stream_batches: int, total_steps: int,
             variant: str, vocab_size: int, model, optimizer, precision) -> None:
        """Write the state after ``step`` completed optimizer steps."""
        self.store.save(
            str(self.path),
            step=step,
            tokens_seen=tokens_seen,
            stream_batches=stream_batches,
            total_steps=total_steps,
            variant=variant,
            vocab_size=vocab_size,
            model_state_dict=model.state_dict(),
            optimizer_state_dict=optimizer.state_dict(),
            scaler_state_dict=precision.state_dict(),
            cpu_rng_state=torch.get_rng_state(),
            cuda_rng_state=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        )

    # ----- restoring -----
    def load(self) -> dict | None:
        """The saved state, or None when there is no resume checkpoint."""
        if not self.path.exists():
            return None
        return self.store.load(str(self.path), map_location="cpu")

    def apply(self, state: dict, *, model, optimizer, precision, variant: str, vocab_size: int) -> None:
        """Put the saved weights, optimizer, loss scale and random state back."""
        # Step 1: refuse a checkpoint from a different model instead of failing deep inside load_state_dict.
        if state["variant"] != variant or state["vocab_size"] != vocab_size:
            raise ValueError(
                f"resume checkpoint is for variant={state['variant']} vocab={state['vocab_size']}, "
                f"this run is variant={variant} vocab={vocab_size}. Use --no-resume to start fresh."
            )
        # Step 2: weights, optimizer state and the fp16 loss scale.
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        precision.load_state_dict(state["scaler_state_dict"])
        # Step 3: random state, so dropout and router noise continue as if nothing had happened.
        torch.set_rng_state(state["cpu_rng_state"])
        cuda_state = state.get("cuda_rng_state")
        if cuda_state is not None and torch.cuda.is_available() and len(cuda_state) == torch.cuda.device_count():
            torch.cuda.set_rng_state_all(cuda_state)

    # ----- finishing -----
    def mark_complete(self, step: int) -> None:
        """Record that the run finished and drop the rolling checkpoint (the final weights are saved elsewhere)."""
        self.complete_path.write_text(json.dumps({"step": step}))
        self.path.unlink(missing_ok=True)

    def completed_step(self) -> int | None:
        """The step at which an earlier run of this folder finished, or None."""
        if not self.complete_path.exists():
            return None
        return int(json.loads(self.complete_path.read_text())["step"])

    def clear(self) -> None:
        """Forget any earlier run (a fresh start must not be mistaken for a resumable one later)."""
        self.path.unlink(missing_ok=True)
        self.complete_path.unlink(missing_ok=True)
        for leftover in self.path.parent.glob("*.tmp"):  # temp files from a save that was interrupted
            os.remove(leftover)
