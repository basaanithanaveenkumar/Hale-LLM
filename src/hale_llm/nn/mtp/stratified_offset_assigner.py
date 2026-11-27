"""Assign the offsets of a whole training step at once, stratified, and hand each micro-batch its share."""

from __future__ import annotations

import torch
from torch import nn

from hale_llm.nn.mtp.offset_share import OffsetShare

MAX_SEED = 2**31 - 1
_MASK64 = (1 << 64) - 1
REMAINDER_MODES = ("balanced", "random")


def _step_seed(seed: int, step: int) -> int:
    """A 32-bit generator seed that depends on BOTH ``seed`` and ``step``.

    ``torch.Generator`` on the CPU only uses the low 32 bits of the seed, so packing the two numbers into one big
    integer would silently drop ``seed``. They are mixed with the SplitMix64 finalizer and 32 bits are kept.
    """
    z = (seed * 0x9E3779B97F4A7C15 + step * 0xBF58476D1CE4E5B9 + 0x94D049BB133111EB) & _MASK64
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _MASK64
    return (z ^ (z >> 31)) & 0xFFFFFFFF


class StratifiedOffsetAssigner(nn.Module):
    """Gives every sequence of a training step an offset in ``1 .. n_offsets``, stratified instead of independent.

    The unit is the **global batch** of one optimizer step: ``micro_batch x accumulation_steps x data-parallel ranks``
    sequences. Drawing an offset per micro-batch would let a micro-batch of 1 see a single offset and would let every
    rank draw on its own. Instead the whole step is planned in one place::

        1. repeat the list 1..N  floor(G / N)  times           G = global batch,  e.g. N = 4, G = 10 -> 1 2 3 4 1 2 3 4
        2. add the remainder: G mod N DISTINCT offsets          -> 1 2 3 4 1 2 3 4  +  3 1
        3. shuffle the whole list                               -> which sequence gets which offset is random
        4. micro-batch m of rank r takes the slice at index (m * ranks + r)

    So within one step every offset is used ``floor(G / N)`` times, plus once more for the remainder offsets. The plan is a
    pure function of ``(seed, step)``, so every rank computes the identical plan without communicating, and no two ranks
    can draw independently. The remainder offsets are chosen by ``remainder``:

    * ``"balanced"`` (default): the offsets that got an extra sequence least often so far, ties broken at random, so
      the totals over the run stay within 1 of each other even when ``G`` is not a multiple of ``N``;
    * ``"random"``: a fresh random subset every step (no memory).

    The position (``step``, ``micro``) and the remainder counts are part of ``state_dict()``, so a resumed run continues
    with exactly the same offsets. The global random state is never touched.
    """

    def __init__(self, n_offsets: int, seed: int = 0, accum_steps: int = 1, remainder: str = "balanced") -> None:
        super().__init__()
        if n_offsets < 1:
            raise ValueError(f"n_offsets={n_offsets} must be at least 1")
        if not 0 <= seed <= MAX_SEED:
            raise ValueError(f"seed={seed} must be between 0 and {MAX_SEED}")
        if accum_steps < 1:
            raise ValueError(f"accum_steps={accum_steps} must be at least 1")
        if remainder not in REMAINDER_MODES:
            raise ValueError(f"remainder={remainder!r} must be one of {REMAINDER_MODES}")
        self.n_offsets = n_offsets
        self.seed = seed
        self.accum_steps = accum_steps
        self.remainder = remainder
        self.step = 0  # optimizer steps planned so far
        self.micro = 0  # micro-batches already handed out in the current step
        self._plan: torch.Tensor | None = None  # the current step's global plan, built on its first micro-batch
        self._batch_size = 0  # sequences per micro-batch of the current step
        self._extra_counts = [0] * n_offsets  # how often each offset received a remainder sequence (balanced mode)
        self._rank: int | None = None  # None = ask torch.distributed when needed
        self._world_size: int | None = None

    # ----- data parallelism -----
    def set_data_parallel(self, rank: int, world_size: int) -> None:
        """Tell the assigner which data-parallel rank this process is (default: read from ``torch.distributed``)."""
        if world_size < 1 or not 0 <= rank < world_size:
            raise ValueError(f"rank={rank} must be in 0..{world_size - 1} for world_size={world_size}")
        self._rank, self._world_size = rank, world_size

    def _data_parallel(self) -> tuple[int, int]:
        if self._world_size is not None:
            return self._rank, self._world_size
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return torch.distributed.get_rank(), torch.distributed.get_world_size()
        return 0, 1

    # ----- handing out offsets -----
    def next_share(self, batch_size: int, device: torch.device | str = "cpu") -> OffsetShare:
        """The offsets of this rank's next micro-batch of ``batch_size`` sequences, in the current step's plan."""
        if batch_size < 1:
            raise ValueError(f"batch_size={batch_size} must be at least 1")
        rank, world = self._data_parallel()
        # 1. First micro-batch of a step: plan the whole global batch (all micro-batches of all ranks) in one go.
        if self._plan is None:
            self._plan = self._build_plan(self.step, batch_size * self.accum_steps * world)
            self._batch_size = batch_size
        elif batch_size != self._batch_size:
            raise ValueError(
                f"the micro-batch size changed inside a step ({self._batch_size} -> {batch_size}); "
                "the offsets of a step are planned for one micro-batch size"
            )
        # 2. This micro-batch is slice number (micro * world + rank) of the plan.
        first = (self.micro * world + rank) * batch_size
        offsets = self._plan[first : first + batch_size]
        global_counts = torch.bincount(self._plan - 1, minlength=self.n_offsets)
        share = OffsetShare(offsets.to(device), global_counts.to(device), self.accum_steps * world)
        # 3. Move on: after the last micro-batch the step is over and the next one gets a new plan.
        self.micro += 1
        if self.micro == self.accum_steps:
            self.step, self.micro, self._plan = self.step + 1, 0, None
        return share

    def _build_plan(self, step: int, global_batch: int) -> torch.Tensor:
        """Offsets of every sequence of the step, ``[global_batch]`` int64, stratified and shuffled."""
        generator = torch.Generator()  # private: the global random state is left alone
        generator.manual_seed(_step_seed(self.seed, step))
        full_rounds, remainder = divmod(global_batch, self.n_offsets)
        # 1. floor(G / N) full copies of 1..N.
        offsets = torch.arange(1, self.n_offsets + 1).repeat(full_rounds)  # [full_rounds * N]
        # 2. The remainder: G mod N distinct offsets.
        offsets = torch.cat([offsets, self._remainder_offsets(remainder, generator)])  # [G]
        # 3. Shuffle, so it is random which sequence gets which offset.
        return offsets[torch.randperm(global_batch, generator=generator)]

    def _remainder_offsets(self, count: int, generator: torch.Generator) -> torch.Tensor:
        """``count`` distinct offsets for the sequences that do not fill a whole round."""
        if count == 0:
            return torch.empty(0, dtype=torch.long)
        if self.remainder == "random":
            return torch.randperm(self.n_offsets, generator=generator)[:count] + 1
        # balanced: least-used first (the counts are whole numbers, the tie-break noise is below 1), ties random.
        noise = torch.rand(self.n_offsets, generator=generator, dtype=torch.float64)
        chosen = (torch.tensor(self._extra_counts, dtype=torch.float64) + noise).argsort()[:count]
        for index in chosen.tolist():
            self._extra_counts[index] += 1
        return chosen + 1

    # ----- saving and resuming -----
    def get_extra_state(self) -> dict:
        """Saved in ``state_dict()`` under ``<name>._extra_state``, so a resumed run continues with the same offsets."""
        return {
            "n_offsets": self.n_offsets, "seed": self.seed, "remainder": self.remainder,
            "accum_steps": self.accum_steps, "step": self.step, "micro": self.micro,
            "batch_size": self._batch_size, "extra_counts": list(self._extra_counts),
            "plan": self._plan.tolist() if self._plan is not None else None,
        }

    def set_extra_state(self, state: dict) -> None:
        """Restore the position; refuses a state that came from a differently configured assigner."""
        if (state["n_offsets"], state["seed"], state["remainder"]) != (self.n_offsets, self.seed, self.remainder):
            raise ValueError(
                f"offset assigner state is for n_offsets={state['n_offsets']} seed={state['seed']} "
                f"remainder={state['remainder']!r}, this assigner has n_offsets={self.n_offsets} seed={self.seed} "
                f"remainder={self.remainder!r}"
            )
        if state["micro"] and state["accum_steps"] != self.accum_steps:
            raise ValueError(
                f"the saved offset plan is part-way through a step of {state['accum_steps']} micro-batches, "
                f"this run uses accum_steps={self.accum_steps}"
            )
        self.step, self.micro, self._batch_size = int(state["step"]), int(state["micro"]), int(state["batch_size"])
        self._extra_counts = [int(count) for count in state["extra_counts"]]
        self._plan = torch.tensor(state["plan"], dtype=torch.long) if state["plan"] is not None else None
