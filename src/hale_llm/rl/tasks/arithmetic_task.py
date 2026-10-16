"""Toy verifiable task: ``"a+b="`` → two-digit sum (``"3+4=" → "07"``)."""

from __future__ import annotations

import random
from dataclasses import dataclass, field


@dataclass
class ArithmeticTask:
    """All ``(a, b)`` with ``0 <= a, b <= max_operand``, split into SFT-seen and unseen facts.

    Small enough to train in seconds on a CPU. The answer is checkable, and a model warm-started on
    only part of the table must *discover* the rest by sampling, which is what online RL is for.
    Answers have a fixed width of two characters, so every completion has the same length.

    Args:
        max_operand: largest operand.
        sft_fraction: fraction of facts used for the supervised warm start.
        seed: shuffling seed of the split.
    """

    max_operand: int = 9
    sft_fraction: float = 0.4
    seed: int = 0
    pairs: list[tuple[int, int]] = field(init=False)

    completion_length = 2

    def __post_init__(self) -> None:
        self.pairs = [
            (a, b) for a in range(self.max_operand + 1) for b in range(self.max_operand + 1)
        ]
        random.Random(self.seed).shuffle(self.pairs)

    @staticmethod
    def prompt(a: int, b: int) -> str:
        return f"{a}+{b}="

    @staticmethod
    def answer(a: int, b: int) -> str:
        return f"{a + b:02d}"

    @property
    def n_sft(self) -> int:
        return max(1, int(round(len(self.pairs) * self.sft_fraction)))

    @property
    def prompts(self) -> list[str]:
        """Every prompt (the online RL prompt pool)."""
        return [self.prompt(a, b) for a, b in self.pairs]

    @property
    def seen_prompts(self) -> list[str]:
        """Prompts whose answers were in the supervised warm start."""
        return self.prompts[: self.n_sft]

    @property
    def unseen_prompts(self) -> list[str]:
        """Prompts never shown with an answer before RL."""
        return self.prompts[self.n_sft :]

    @property
    def answers(self) -> dict[str, str]:
        return {self.prompt(a, b): self.answer(a, b) for a, b in self.pairs}

    @property
    def sft_texts(self) -> list[str]:
        """``prompt + answer`` strings for the warm start (equal length)."""
        return [self.prompt(a, b) + self.answer(a, b) for a, b in self.pairs[: self.n_sft]]

    @property
    def candidates(self) -> list[str]:
        """Every well-formed answer, ``"00"`` … ``"99"`` (for answer distributions)."""
        return [f"{n:02d}" for n in range(100)]
