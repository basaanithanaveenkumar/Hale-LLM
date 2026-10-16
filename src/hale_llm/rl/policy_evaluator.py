"""Greedy-decoding accuracy of a policy on a task (the number RL is supposed to move)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch


class PolicyEvaluator:
    """Decodes every prompt at near-zero temperature and compares with the reference answer.

    Args:
        tokenizer: char tokenizer.
        variant: model family (selects the registered sampler).
        completion_length: tokens to generate.
        sampling_steps: denoising steps for diffusion variants.
    """

    def __init__(
        self, tokenizer, variant: str, completion_length: int, sampling_steps: int = 2
    ) -> None:
        from hale_core.registry import get_sampler

        self.tokenizer = tokenizer
        self.variant = variant
        self.completion_length = completion_length
        self.sampling_steps = sampling_steps
        self.sampler = get_sampler(variant)

    @torch.no_grad()
    def decode(
        self, model: torch.nn.Module, prompts: Sequence[str], temperature: float = 1e-4
    ) -> list[str]:
        """One completion per prompt."""
        ids = torch.tensor([self.tokenizer.encode(p) for p in prompts])
        kwargs = {"sampling_steps": self.sampling_steps} if self.variant != "autoregressive" else {}
        out = self.sampler(
            model, self.tokenizer, ids, self.completion_length, temperature=temperature, **kwargs
        )
        return [self.tokenizer.decode(row[ids.shape[1] :]) for row in out]

    def accuracy(
        self, model: torch.nn.Module, prompts: Sequence[str], answers: Mapping[str, str]
    ) -> float:
        """Fraction of ``prompts`` decoded exactly right (0 for an empty list)."""
        if not prompts:
            return 0.0
        decoded = self.decode(model, prompts)
        return sum(d == answers[p] for p, d in zip(prompts, decoded, strict=True)) / len(prompts)
