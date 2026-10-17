"""Convenience entry: config → checkpoint → text."""

from __future__ import annotations

from hale_llm.config.schema import RunConfig
from hale_llm.inference.session import GenerationResult, load_session


def generate(
    cfg: RunConfig,
    prompt: str | None = None,
    *,
    checkpoint: str | None = None,
    return_history: bool = False,
    **kwargs,
) -> GenerationResult:
    """Load the session for ``cfg`` and generate one completion of ``prompt``.

    Args:
        cfg: the run config.
        prompt: text to complete (default ``sample.prompt``).
        checkpoint: checkpoint path (default ``train.checkpoint_path``).
        return_history: also return the sampler's step-by-step history.
        **kwargs: forwarded to :meth:`InferenceSession.generate`.
    """
    return load_session(cfg, checkpoint).generate(prompt, return_history=return_history, **kwargs)
