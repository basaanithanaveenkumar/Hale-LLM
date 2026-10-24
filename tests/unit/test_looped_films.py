"""Looped films: docstring corpus, LoopLab traces, every film renders, variant configs."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from hale_core.nn.stacks import LOOP_PRESETS, LoopSpec
from PIL import Image

from hale_llm.config.schema import load_config
from hale_llm.visualization.explainers.canvas import GifWriter
from hale_llm.visualization.looped_films import LOOP_FILMS, LoopLab, docstring_corpus

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def lab() -> LoopLab:
    return LoopLab(steps=3, text_steps=3)


def test_docstring_corpus_is_real_text() -> None:
    text = docstring_corpus(alphabet="abcdefghijklmnopqrstuvwxyz ")
    assert len(text) > 10_000 and set(text) <= set("abcdefghijklmnopqrstuvwxyz ")
    assert "  " not in docstring_corpus()[:2000]


def test_trace_records_every_pass(lab: LoopLab) -> None:
    run = lab.train("universal_transformer", n_loops=3)
    trace = lab.trace(run)
    n = len(trace.text)
    assert trace.n_passes == trace.loops == 3
    assert all(s.shape == (n, 64) for s in trace.states)
    assert all(
        ids.shape == (n, 3) and p.shape == (n, 3)
        for ids, p in zip(trace.lens_ids, trace.lens_probs)
    )
    assert torch.allclose(trace.change[0], torch.ones(n)) and len(trace.residuals()) == 3
    assert lab.train("universal_transformer", n_loops=3) is run  # cached


def test_act_and_mor_traces_carry_per_token_depth(lab: LoopLab) -> None:
    act = lab.trace(lab.train("universal_transformer_act"))
    assert act.depth is not None and len(act.halting) == act.n_passes == len(act.running)
    assert act.depth.min() >= 1 and act.depth.max() <= act.n_passes
    mor = lab.trace(lab.train("mixture_of_recursions"))
    assert mor.depth is not None and len(mor.running) == mor.n_passes
    assert all(bool(r.eq(mor.depth > t).all()) for t, r in enumerate(mor.running))


def test_text_lm_and_loop_budget(lab: LoopLab) -> None:
    run = lab.train_text({"arch": "looped", "n_layers": 1, "loop": {"n_loops": 2}})
    assert len(run.losses) == 3
    bpc = lab.text_loss(run, n_loops=3, batches=2)
    assert (
        bpc > 0 and bpc == bpc
    )  # finite (3 steps leave the large initial loss of tied embeddings)


@pytest.mark.parametrize("name", sorted(LOOP_FILMS.keys()))
def test_every_film_renders(name: str, lab: LoopLab) -> None:
    writer = GifWriter(dpi=20, colors=16)
    LOOP_FILMS.get(name)().render(lab, writer)
    assert len(writer) > 10 and writer.seconds > 3


def test_film_build_writes_a_gif(lab: LoopLab, tmp_path: Path) -> None:
    path = LOOP_FILMS.get("sharing_patterns")().build(tmp_path / "s.gif", lab)
    with Image.open(path) as gif:
        assert gif.is_animated


@pytest.mark.parametrize("name", sorted(LOOP_PRESETS))
def test_variant_config_is_exactly_the_preset(name: str) -> None:
    cfg = load_config(ROOT / "configs" / "looped" / "variants" / f"{name}.yaml")
    assert cfg.variant == "looped" and cfg.model.arch == "looped"
    assert LoopSpec.coerce(cfg.model.loop) == LoopSpec.preset_spec(name)
