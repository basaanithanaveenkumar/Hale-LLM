"""Animated explainers of every generation paradigm in ``hale_llm.models``.

``hale-llm-explain`` (or :func:`build_explainer_gifs`) trains a tiny model of each variant on a
four-sentence character corpus and renders a two-act GIF: how a training example is built, then
generation step by step with the model's real probabilities. The GIFs ship in
``assets/explainers/`` and are embedded in the README and ``notebooks/models/``.

Modules: ``canvas`` (drawing kit + GIF writer), ``tokenizer`` (character tokens), ``toy`` (toy
training), ``base`` (contract + registry), ``masked`` / ``blocks`` (shared masked-token and
block panels) and one module per paradigm.
"""

from __future__ import annotations

from pathlib import Path

from hale_llm.visualization.explainers import autoregressive as _ar  # noqa: F401
from hale_llm.visualization.explainers import block_diffusion as _bd  # noqa: F401
from hale_llm.visualization.explainers import block_flow_matching as _bfm  # noqa: F401
from hale_llm.visualization.explainers import diffusion as _diff  # noqa: F401
from hale_llm.visualization.explainers import flow_matching as _fm  # noqa: F401
from hale_llm.visualization.explainers import looped as _looped  # noqa: F401
from hale_llm.visualization.explainers import mtp as _mtp  # noqa: F401
from hale_llm.visualization.explainers.base import (
    CORPUS,
    EXPLAINERS,
    SENTENCE,
    Explainer,
    get_explainer,
    register_explainer,
)
from hale_llm.visualization.explainers.canvas import GifWriter
from hale_llm.visualization.explainers.tokenizer import CharTokenizer
from hale_llm.visualization.explainers.toy import ToyRun, train_toy

#: README / gallery order.
ORDER = (
    "autoregressive",
    "mtp",
    "looped",
    "diffusion",
    "flow_matching",
    "block_diffusion",
    "block_flow_matching",
)


def build_explainer_gifs(
    out_dir: str | Path = "assets/explainers", variants=None, seed: int = 0
) -> list[Path]:
    """Train, trace and render the explainer GIF of each variant into ``out_dir/<variant>.gif``."""
    out = []
    for name in variants or ORDER:
        out.append(get_explainer(name).build(Path(out_dir) / f"{name}.gif", seed=seed))
    return out


__all__ = [
    "CORPUS",
    "EXPLAINERS",
    "ORDER",
    "SENTENCE",
    "CharTokenizer",
    "Explainer",
    "GifWriter",
    "ToyRun",
    "build_explainer_gifs",
    "get_explainer",
    "register_explainer",
    "train_toy",
]
