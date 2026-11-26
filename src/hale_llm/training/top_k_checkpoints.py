"""Keep only the N best checkpoints of a run, ranked by a validation score."""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any

from loguru import logger

from hale_llm.runtime import CheckpointStore


class TopKCheckpoints:
    """Save a checkpoint when its score ranks in the top ``keep``; delete the one it pushes out.

    ``score_name`` is only used in file names (``step0000500_loss_1.2345.pt``).
    ``higher_is_better`` is False for a loss and True for an accuracy.
    """

    def __init__(self, directory: str | Path, keep: int, score_name: str, higher_is_better: bool,
                 store: CheckpointStore | None = None) -> None:
        self.directory = Path(directory)
        self.keep = keep
        self.score_name = score_name
        self.higher_is_better = higher_is_better
        self.store = store or CheckpointStore()
        self._kept: list[tuple[float, Path]] = []  # best first

    def _rank_key(self, score: float) -> float:
        """Larger key = better, whatever the direction of the score."""
        return score if self.higher_is_better else -score

    def would_keep(self, score: float) -> bool:
        """True if ``score`` earns a place: a free slot, or better than the worst kept one."""
        if self.keep <= 0 or not math.isfinite(score):
            return False
        if len(self._kept) < self.keep:
            return True
        return self._rank_key(score) > self._rank_key(self._kept[-1][0])

    def offer(self, score: float, step: int, **state: Any) -> Path | None:
        """Save ``state`` if ``score`` ranks in the top ``keep``. Returns the path, or None."""
        # Step 1: decide whether this score earns a place.
        if not self.would_keep(score):
            return None
        # Step 2: write the checkpoint, named so the folder listing reads like a leaderboard.
        path = self.directory / f"step{step:07d}_{self.score_name}_{score:.4f}.pt"
        self.store.save(str(path), step=step, **{self.score_name: score}, **state)
        # Step 3: insert it in rank order and delete whatever fell out of the top ``keep``.
        self._kept.append((score, path))
        self._kept.sort(key=lambda item: self._rank_key(item[0]), reverse=True)
        while len(self._kept) > self.keep:
            _, evicted = self._kept.pop()
            evicted.unlink(missing_ok=True)
            logger.info("dropped checkpoint {} (no longer in the top {})", evicted.name, self.keep)
        return path

    def restore_from_disk(self, max_step: int) -> None:
        """Rebuild the leaderboard from the files already in the folder (used when resuming a run).

        The score and step are read back from the file names. Files from steps after ``max_step``
        belong to the part of the run that was interrupted and will be redone, so they are deleted.
        """
        pattern = re.compile(rf"step(\d+)_{re.escape(self.score_name)}_(-?\d+\.\d+)\.pt$")
        self._kept = []
        for path in sorted(self.directory.glob("*.pt")):
            match = pattern.match(path.name)
            if match is None:
                continue
            if int(match.group(1)) > max_step:
                path.unlink()
                logger.info("removed {} (from after the resume point)", path.name)
                continue
            self._kept.append((float(match.group(2)), path))
        self._kept.sort(key=lambda item: self._rank_key(item[0]), reverse=True)
        while len(self._kept) > self.keep:  # keep_best may have been lowered since the last run
            self._kept.pop()[1].unlink(missing_ok=True)
        logger.info("restored {} best checkpoints from {}", len(self._kept), self.directory)

    @property
    def paths(self) -> list[Path]:
        """Kept checkpoints, best first."""
        return [path for _, path in self._kept]
