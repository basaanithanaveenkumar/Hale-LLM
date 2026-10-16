"""Explainer animations: tokenizer, toy training, trace fidelity to the samplers, rendering."""

from __future__ import annotations

import pytest
import torch
from PIL import Image

from hale_llm.visualization.explainers import (
    CORPUS,
    EXPLAINERS,
    ORDER,
    CharTokenizer,
    GifWriter,
    get_explainer,
    train_toy,
)
from hale_llm.visualization.explainers.canvas import Layout, Stage
from hale_llm.visualization.explainers.sequential import candidate_items

import hale_llm.models  # noqa: F401,E402

MODELS = {"autoregressive", "mtp", "looped", "diffusion", "flow_matching", "block_diffusion", "block_flow_matching"}


def test_every_model_package_has_an_explainer() -> None:
    from hale_llm.core.registry import MODELS as REGISTERED

    assert set(EXPLAINERS) == set(ORDER) == MODELS <= set(REGISTERED)


def test_char_tokenizer_round_trip() -> None:
    tok = CharTokenizer()
    ids = tok(CORPUS[0], return_tensors="pt")["input_ids"]
    assert ids.shape == (1, 48) and tok.decode(ids[0]) == CORPUS[0]
    padded = tok("ab", padding="max_length", max_length=5)["input_ids"]
    assert padded[2:] == [tok.pad_token_id] * 3 and tok.decode(padded) == "ab"
    assert tok.token(tok.mask_token_id) == "<mask>" and len(tok) == tok.vocab_size
    with pytest.raises(ValueError, match="outside"):
        tok.encode("é")


def test_corpus_has_branch_points_and_one_canvas() -> None:
    assert len({len(t) for t in CORPUS}) == 1 and len(CORPUS[0]) % 8 == 0
    branch = [i for i in range(48) if len({t[i] for t in CORPUS}) > 1]
    assert branch and min(branch) >= 8  # the 8-char prompt is shared
    with pytest.raises(ValueError, match="equal length"):
        train_toy("autoregressive", ["abc", "abcd"], steps=1)


@pytest.mark.parametrize("variant", sorted(MODELS))
def test_trace_reproduces_the_registered_sampler(variant: str) -> None:
    """With the same seed an explainer's trace emits exactly the sampler's tokens."""
    explainer = get_explainer(variant)
    kwargs = dict(explainer.train_kwargs)
    kwargs["steps"] = 3  # behaviour of an untrained model is just as deterministic
    run = train_toy(variant, explainer.texts, **kwargs)
    trace = explainer.trace(run, seed=4)
    assert torch.equal(trace.ids, explainer.reference(run, seed=4))
    assert trace.ids.size(1) == 48


@pytest.mark.parametrize("variant", ["autoregressive", "block_flow_matching", "mtp", "looped"])
def test_render_writes_an_animated_gif(variant: str, tmp_path) -> None:
    explainer = get_explainer(variant)
    kwargs = dict(explainer.train_kwargs, steps=2)
    run = train_toy(variant, explainer.texts, **kwargs)
    writer = GifWriter(dpi=30, colors=32)
    explainer.render(run, explainer.trace(run), writer)
    path = writer.save(tmp_path / f"{variant}.gif")
    with Image.open(path) as gif:
        assert gif.is_animated and 10 < gif.n_frames <= len(writer)  # Pillow merges identical frames
        assert gif.size == writer.frames[0].size and gif.size[0] == 300  # 10 in × 30 dpi, 16:9


def test_layout_rows_blocks_and_writer_hold(tmp_path) -> None:
    layout = Layout(48, block=8)
    assert layout.rows == 2 and layout.pos(24)[1] < layout.pos(23)[1]  # second row is lower
    assert layout.pos(8)[0] - layout.pos(7)[0] > layout.step + 1.0  # gap between blocks
    x0, _, x1, _ = layout.block_span(1)
    assert x0 < layout.pos(8)[0] < layout.pos(15)[0] < x1
    writer = GifWriter(dpi=20)
    writer.stage(Stage("title", "subtitle", "phase", "caption"), 100)
    writer.hold(400)
    assert writer.durations == [500] and writer.seconds == 0.5
    with pytest.raises(ValueError):
        GifWriter().save(tmp_path / "empty.gif")


def test_candidate_items_drop_noise_but_keep_the_sample() -> None:
    from types import SimpleNamespace

    run = SimpleNamespace(tokenizer=CharTokenizer())
    tok = run.tokenizer
    a, b, c = (tok.stoi[ch] for ch in "abc")
    items, chosen = candidate_items(run, SimpleNamespace(top_ids=[a, b, c], top_probs=[0.97, 0.02, 0.01], chosen=a))
    assert items == [("a", 0.97)] and chosen == 0
    items, chosen = candidate_items(run, SimpleNamespace(top_ids=[a, b, c], top_probs=[0.97, 0.02, 0.01], chosen=c))
    assert [ch for ch, _ in items] == ["a", "c"] and chosen == 1  # an unlikely draw is still shown
