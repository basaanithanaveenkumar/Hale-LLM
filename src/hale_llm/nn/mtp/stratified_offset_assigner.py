"""Assign the offsets of a whole training step at once, stratified, and hand each micro-batch its share."""

from __future__ import annotations

import itertools

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
    """Gives every sequence of a training step ``offsets_per_sequence`` offsets of ``1 .. n_offsets``, stratified.

    The unit is the **global batch** of one optimizer step: ``micro_batch x accumulation_steps x data-parallel ranks``
    sequences. Drawing an offset per micro-batch would let a micro-batch of 1 see a single offset and would let every
    rank draw on its own. Instead the whole step is planned in one place::

        1. repeat the list 1..N  floor(G / N)  times           G = global batch,  e.g. N = 4, G = 10 -> 1 2 3 4 1 2 3 4
        2. add the remainder: G mod N DISTINCT offsets          -> 1 2 3 4 1 2 3 4  +  3 1
        3. shuffle the whole list                               -> which sequence gets which offset is random
        4. micro-batch m of rank r takes the slice at index (m * ranks + r)

    So within one step every offset is used ``floor(G / N)`` times, plus once more for the remainder offsets.

    With ``offsets_per_sequence = K > 1`` a sequence gets a *combination* of K distinct offsets (K = 2 of N = 4: one of
    ``12 13 14 23 24 34``). The same recipe is applied to the ``C = N choose K`` combinations instead of the offsets::

        1. repeat the list of all C combinations  floor(G / C)  times     every offset is in the same number of them
        2. add the remainder: G mod C DISTINCT combinations, chosen so the offsets stay balanced (see below)
        3. shuffle, and hand out the slices as above

    so every offset AND every combination is used equally often when ``G`` is a multiple of ``C``. With
    ``stratified=False`` each sequence instead draws its K offsets independently (uniform, no repeats), which is only
    balanced in expectation. The plan is a
    pure function of ``(seed, step)``, so every rank computes the identical plan without communicating, and no two ranks
    can draw independently. The remainder offsets are chosen by ``remainder``:

    * ``"balanced"`` (default): the offsets that got an extra sequence least often so far (for K > 1: the combinations
      whose offsets did, then the least used combinations), ties broken at random, so the totals over the run stay
      within about 1 of each other even when ``G`` is not a multiple of ``C``;
    * ``"random"``: a fresh random subset every step (no memory).

    The position (``step``, ``micro``) and the remainder counts are part of ``state_dict()``, so a resumed run continues
    with exactly the same offsets. The global random state is never touched.
    """

    def __init__(
        self,
        n_offsets: int,
        seed: int = 0,
        accum_steps: int = 1,
        remainder: str = "balanced",
        offsets_per_sequence: int = 1,
        stratified: bool = True,
    ) -> None:
        super().__init__()
        if n_offsets < 1:
            raise ValueError(f"n_offsets={n_offsets} must be at least 1")
        if not 0 <= seed <= MAX_SEED:
            raise ValueError(f"seed={seed} must be between 0 and {MAX_SEED}")
        if accum_steps < 1:
            raise ValueError(f"accum_steps={accum_steps} must be at least 1")
        if remainder not in REMAINDER_MODES:
            raise ValueError(f"remainder={remainder!r} must be one of {REMAINDER_MODES}")
        if not 1 <= offsets_per_sequence <= n_offsets:
            raise ValueError(f"offsets_per_sequence={offsets_per_sequence} must be between 1 and n_offsets={n_offsets}")
        self.n_offsets = n_offsets
        self.offsets_per_sequence = offsets_per_sequence
        self.stratified = stratified
        # Every way to pick K distinct offsets of 1..N, [C, K]; with K = 1 these are just the offsets.
        # Built from plain Python and pinned to the CPU: the dry run constructs the model under a "meta" default device,
        # where torch.combinations has no kernel (and a plan living on "meta" could not be indexed).
        combos = list(itertools.combinations(range(1, n_offsets + 1), offsets_per_sequence))
        self._combos = torch.tensor(combos, dtype=torch.long, device="cpu")
        self.seed = seed
        self.accum_steps = accum_steps
        self.remainder = remainder
        self.step = 0  # optimizer steps planned so far
        self.micro = 0  # micro-batches already handed out in the current step
        self._plan: torch.Tensor | None = None  # the current step's global plan, built on its first micro-batch
        self._batch_size = 0  # sequences per micro-batch of the current step
        self._extra_counts = [0] * n_offsets  # how often each offset was in a remainder sequence (balanced mode)
        self._combo_counts = [0] * self._combos.size(0)  # how often each combination was a remainder one
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
        offsets = self._plan[first : first + batch_size]  # [B, K]
        global_counts = torch.bincount(self._plan.flatten() - 1, minlength=self.n_offsets)
        share = OffsetShare(offsets.to(device), global_counts.to(device), self.accum_steps * world)
        # 3. Move on: after the last micro-batch the step is over and the next one gets a new plan.
        self.micro += 1
        if self.micro == self.accum_steps:
            self.step, self.micro, self._plan = self.step + 1, 0, None
        return share

    def _build_plan(self, step: int, global_batch: int) -> torch.Tensor:
        """Offsets of every sequence of the step, ``[global_batch, K]`` int64, stratified and shuffled."""
        generator = torch.Generator()  # private: the global random state is left alone
        generator.manual_seed(_step_seed(self.seed, step))
        if not self.stratified:
            # Independent draws: K distinct offsets per sequence, uniformly (the K smallest of N random numbers).
            random_scores = torch.rand(global_batch, self.n_offsets, generator=generator)
            return random_scores.argsort(dim=1)[:, : self.offsets_per_sequence] + 1
        combos_in_a_round = self._combos.size(0)
        full_rounds, remainder = divmod(global_batch, combos_in_a_round)
        # 1. floor(G / C) full copies of the list of combinations (K = 1: of 1..N).
        plan = self._combos.repeat(full_rounds, 1)  # [full_rounds * C, K]
        # 2. The remainder: G mod C distinct combinations.
        plan = torch.cat([plan, self._remainder_combos(remainder, generator)])  # [G, K]
        # 3. Shuffle, so it is random which sequence gets which combination.
        return plan[torch.randperm(global_batch, generator=generator)]

    def _remainder_combos(self, count: int, generator: torch.Generator) -> torch.Tensor:
        """``count`` distinct combinations for the sequences that do not fill a whole round, ``[count, K]``."""
        n_combos = self._combos.size(0)
        if count == 0:
            return self._combos[:0]
        if self.remainder == "random":
            return self._combos[torch.randperm(n_combos, generator=generator)[:count]]
        # balanced: pick one combination at a time, the one whose offsets are the least loaded so far (whole numbers), then
        # (K > 1) the least used combination, then a random one: the tie-break noise stays below 1.
        noise = torch.rand(n_combos, generator=generator, dtype=torch.float64)
        load = torch.tensor(self._extra_counts, dtype=torch.float64)  # [N]
        members = self._combos - 1  # [C, K] offset indices of each combination
        combo_uses = torch.tensor(self._combo_counts, dtype=torch.float64)  # [C]
        load_weight = combo_uses.max() + 2.0 if self.offsets_per_sequence > 1 else 1.0  # load outranks combination use
        combo_term = combo_uses if self.offsets_per_sequence > 1 else torch.zeros_like(combo_uses)  # K = 1: load says it all
        taken = torch.zeros(n_combos, dtype=torch.bool)
        chosen = []
        for _ in range(count):
            score = load[members].sum(dim=1) * load_weight + combo_term + noise  # [C]
            score[taken] = float("inf")  # distinct combinations only
            index = int(score.argmin())
            chosen.append(index)
            taken[index] = True
            load[members[index]] += 1  # the next pick sees this offset as one more used
        for index in chosen:
            self._combo_counts[index] += 1
            for offset_index in members[index].tolist():
                self._extra_counts[offset_index] += 1
        return self._combos[chosen]

    # ----- saving and resuming -----
    def get_extra_state(self) -> dict:
        """Saved in ``state_dict()`` under ``<name>._extra_state``, so a resumed run continues with the same offsets."""
        return {
            "n_offsets": self.n_offsets, "seed": self.seed, "remainder": self.remainder,
            "offsets_per_sequence": self.offsets_per_sequence, "stratified": self.stratified,
            "accum_steps": self.accum_steps, "step": self.step, "micro": self.micro,
            "batch_size": self._batch_size, "extra_counts": list(self._extra_counts),
            "combo_counts": list(self._combo_counts),
            "plan": self._plan.tolist() if self._plan is not None else None,
        }

    def set_extra_state(self, state: dict) -> None:
        """Restore the position; refuses a state that came from a differently configured assigner."""
        saved = (
            state["n_offsets"], state["seed"], state["remainder"],
            state.get("offsets_per_sequence", 1), state.get("stratified", True),
        )
        mine = (self.n_offsets, self.seed, self.remainder, self.offsets_per_sequence, self.stratified)
        if saved != mine:
            raise ValueError(
                f"offset assigner state is for (n_offsets, seed, remainder, offsets_per_sequence, stratified) = {saved}, "
                f"this assigner has {mine}"
            )
        if state["micro"] and state["accum_steps"] != self.accum_steps:
            raise ValueError(
                f"the saved offset plan is part-way through a step of {state['accum_steps']} micro-batches, "
                f"this run uses accum_steps={self.accum_steps}"
            )
        self.step, self.micro, self._batch_size = int(state["step"]), int(state["micro"]), int(state["batch_size"])
        self._extra_counts = [int(count) for count in state["extra_counts"]]
        self._combo_counts = [int(count) for count in state.get("combo_counts", self._combo_counts)]
        self._plan = torch.tensor(state["plan"], dtype=torch.long) if state["plan"] is not None else None
