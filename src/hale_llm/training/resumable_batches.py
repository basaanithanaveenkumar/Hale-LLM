"""An endless stream of training batches that can continue from any position."""

from __future__ import annotations

from collections.abc import Iterator

import torch
from torch.utils.data import DataLoader


class ResumableBatches:
    """Yields micro-batches forever, in an order that depends only on ``(seed, epoch)``.

    A plain ``DataLoader`` reshuffles from the global random state, so after an interruption there is
    no way to know which batch came next. Here every epoch gets its own seeded shuffle, so the
    position of a run is just one number (``consumed``). Starting from ``start_batch`` skips by
    index: the skipped batches are never loaded or collated, so resuming is instant even deep into
    an epoch.
    """

    def __init__(self, loader: DataLoader, seed: int, start_batch: int = 0) -> None:
        self.loader = loader
        self.seed = seed
        self.batches_per_epoch = len(loader)
        if self.batches_per_epoch < 1:
            raise ValueError("the training loader has no batches")
        self.consumed = (
            start_batch  # micro-batches handed out so far (saved in the resume checkpoint)
        )
        self._batches = self._generate(start_batch)

    def __iter__(self) -> Iterator[dict]:
        return self

    def __next__(self) -> dict:
        batch = next(self._batches)
        self.consumed += 1
        return batch

    def _epoch_index_batches(self, epoch: int) -> list[list[int]]:
        """The dataset indices of every batch of one epoch, shuffled by a generator seeded from the epoch."""
        sampler = self.loader.batch_sampler.sampler
        if hasattr(
            sampler, "generator"
        ):  # a RandomSampler; a sequential one is already deterministic
            sampler.generator = torch.Generator().manual_seed(self.seed * 1_000_003 + epoch)
        return list(self.loader.batch_sampler)

    def _generate(self, start_batch: int) -> Iterator[dict]:
        # Step 1: which epoch the run is in, and how many of its batches were already used.
        epoch, skip = divmod(start_batch, self.batches_per_epoch)
        while True:
            # Step 2: this epoch's batches, minus the ones already consumed (cheap: indices only).
            remaining = self._epoch_index_batches(epoch)[skip:]
            # Step 3: load the data for the remaining batches only.
            epoch_loader = DataLoader(
                self.loader.dataset,
                batch_sampler=remaining,
                collate_fn=self.loader.collate_fn,
                num_workers=self.loader.num_workers,
                pin_memory=self.loader.pin_memory,
                # Creating a DataLoader iterator draws a random number. Give it its own generator so
                # that draw never comes from the global one (which dropout and router noise use):
                # otherwise a resumed run would be one draw off from an uninterrupted one.
                generator=torch.Generator().manual_seed(self.seed * 1_000_003 + epoch),
            )
            yield from epoch_loader
            epoch, skip = epoch + 1, 0
