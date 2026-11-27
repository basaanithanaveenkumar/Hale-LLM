"""Random offsets that are used equally often over the whole run."""

from __future__ import annotations

import torch
from torch import nn

MAX_SEED = 2**31 - 1
_MASK64 = (1 << 64) - 1


def _deck_seed(seed: int, deck: int) -> int:
    """A 32-bit generator seed that depends on BOTH ``seed`` and the deck number.

    ``torch.Generator`` on the CPU only uses the low 32 bits of the seed it is given, so packing the two numbers into
    one big integer (``seed * 2**32 + deck``) would silently drop ``seed``. Instead they are mixed with the SplitMix64
    finalizer, which spreads every input bit over the whole word, and the low 32 bits are kept.
    """
    z = (seed * 0x9E3779B97F4A7C15 + deck * 0xBF58476D1CE4E5B9 + 0x94D049BB133111EB) & _MASK64
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _MASK64
    return (z ^ (z >> 31)) & 0xFFFFFFFF


class BalancedOffsetSampler(nn.Module):
    """Draws offsets ``1 .. n_offsets`` in a random order that still gives every offset the same frequency.

    Picking each offset independently at random would make the counts only *roughly* equal. This sampler uses a
    shuffled deck instead:

        deck 0 = a random permutation of 1..N      (for N = 4, e.g. 3 1 4 2)
        deck 1 = another random permutation        (e.g.          2 4 1 3)
        ...

    The draws are the decks laid end to end, so:

    * every block of N consecutive draws (aligned to a deck) contains each offset exactly once;
    * after ANY number of draws the counts of two offsets differ by at most 1;
    * the order inside a deck is random, and different seeds give different orders.

    The only state is ``cursor``, how many offsets were drawn so far. Deck ``b`` is a pure function of
    ``(seed, b)``, so the stream does not depend on how the draws were split into batches, never touches the global
    random-number generator, and resumes exactly from a saved cursor (the cursor is part of ``state_dict()``).
    """

    def __init__(self, n_offsets: int, seed: int = 0) -> None:
        super().__init__()
        if n_offsets < 1:
            raise ValueError(f"n_offsets={n_offsets} must be at least 1")
        if not 0 <= seed <= MAX_SEED:
            raise ValueError(f"seed={seed} must be between 0 and {MAX_SEED}")
        self.n_offsets = n_offsets
        self.seed = seed
        self.cursor = 0  # offsets drawn so far

    # ----- sampling -----
    def sample(self, count: int, device: torch.device | str = "cpu") -> torch.Tensor:
        """The next ``count`` offsets as a ``[count]`` int64 tensor with values in ``1 .. n_offsets``."""
        if count < 0:
            raise ValueError(f"count={count} cannot be negative")
        if count == 0:
            return torch.empty(0, dtype=torch.long, device=device)
        # 1. Which decks the draws ``cursor .. cursor + count - 1`` fall into, and where the first one starts.
        first_deck, start = divmod(self.cursor, self.n_offsets)
        last_deck = (self.cursor + count - 1) // self.n_offsets
        # 2. Lay those decks end to end and cut out exactly the draws we need.
        decks = torch.cat([self._deck(deck) for deck in range(first_deck, last_deck + 1)])  # [n_decks * N]
        offsets = decks[start : start + count]  # [count]
        # 3. Move the cursor, so the next call continues where this one stopped.
        self.cursor += count
        return offsets.to(device)

    def _deck(self, index: int) -> torch.Tensor:
        """Deck number ``index``: a random permutation of ``1 .. n_offsets`` that depends only on ``(seed, index)``."""
        generator = torch.Generator()  # a private generator: the global random state is left alone
        generator.manual_seed(_deck_seed(self.seed, index))
        return torch.randperm(self.n_offsets, generator=generator) + 1

    # ----- saving and resuming -----
    def get_extra_state(self) -> dict:
        """Saved in ``state_dict()`` under ``<name>._extra_state``, so a resumed run continues the same stream."""
        return {"cursor": self.cursor, "n_offsets": self.n_offsets, "seed": self.seed}

    def set_extra_state(self, state: dict) -> None:
        """Restore the stream position; refuses a state that came from a different sampler."""
        if state["n_offsets"] != self.n_offsets or state["seed"] != self.seed:
            raise ValueError(
                f"offset sampler state is for n_offsets={state['n_offsets']} seed={state['seed']}, "
                f"this sampler has n_offsets={self.n_offsets} seed={self.seed}"
            )
        self.cursor = int(state["cursor"])
