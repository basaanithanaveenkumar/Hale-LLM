"""Looped transformers (``model.arch: looped``) across every paradigm, the trainer and inference."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from hale_core.nn.backbones import LoopedBackbone
from hale_core.nn.stacks import LoopSpec, loop_stats

import hale_llm.models  # noqa: F401
from hale_llm.config.schema import load_config
from hale_llm.core.registry import get_loss, get_variant, instantiate_metrics
from hale_llm.dataloader.collate import get_collate
from hale_llm.inference.session import session_from_model
from hale_llm.losses import with_auxiliary_losses
from hale_llm.metrics.base import MetricContext

REPO = Path(__file__).resolve().parents[2]
LOOPED = sorted((REPO / "configs" / "looped").glob("*.yaml"))


def shrink(cfg):
    cfg.device = "cpu"
    cfg.model.d_model, cfg.model.n_heads, cfg.model.d_ff, cfg.model.max_length = 32, 4, 64, 32
    if cfg.model.block_size:
        cfg.model.block_size = 8
    return cfg


def batch_for(cfg, tokenizer, seq: int = 16) -> dict:
    ids = torch.randint(
        2, tokenizer.vocab_size, (2, seq), generator=torch.Generator().manual_seed(0)
    )
    ids[1, -4:] = tokenizer.pad_token_id
    return get_collate(cfg.variant)(ids, tokenizer, cfg=cfg)


def test_looped_configs_exist_for_every_paradigm_with_a_token_head() -> None:
    variants = {load_config(p).variant for p in LOOPED}
    assert {"autoregressive", "diffusion", "block_diffusion"} <= variants
    assert all(load_config(p).model.arch == "looped" for p in LOOPED)


@pytest.mark.parametrize("path", LOOPED, ids=lambda p: p.stem)
def test_every_looped_config_trains_one_step(path: Path, tiny_tokenizer) -> None:
    cfg = shrink(load_config(path))
    torch.manual_seed(0)
    model = get_variant(cfg.variant)(vocab_size=tiny_tokenizer.vocab_size, cfg=cfg)
    assert isinstance(model.backbone, LoopedBackbone)
    assert model.backbone.loop == LoopSpec.coerce(cfg.model.loop)
    loss, extras = with_auxiliary_losses(
        model, get_loss(cfg.variant)(model, batch_for(cfg, tiny_tokenizer))
    )
    assert torch.isfinite(loss)
    assert "loop/loops" in extras
    assert ("ponder_loss" in extras) == (cfg.model.loop.halting == "act")
    loss.backward()
    assert all(p.grad is not None for p in model.backbone.stack.core.parameters())


def test_the_act_ponder_cost_is_added_to_the_task_loss(tiny_tokenizer) -> None:
    cfg = shrink(load_config(REPO / "configs" / "looped" / "autoregressive_act.yaml"))
    model = get_variant(cfg.variant)(vocab_size=tiny_tokenizer.vocab_size, cfg=cfg)
    task = get_loss(cfg.variant)(model, batch_for(cfg, tiny_tokenizer))
    total, extras = with_auxiliary_losses(model, task)
    ponder = model.backbone.stack.aux_loss
    torch.testing.assert_close(total, task + ponder)
    assert extras["task_loss"] == pytest.approx(float(task.detach())) and extras["ponder_loss"] > 0
    assert 1.0 <= extras["loop/mean_updates"] <= cfg.model.loop.n_loops


def test_non_looped_models_have_no_auxiliary_losses(tiny_tokenizer) -> None:
    cfg = shrink(load_config(REPO / "configs" / "autoregressive" / "nano.yaml"))
    model = get_variant(cfg.variant)(vocab_size=tiny_tokenizer.vocab_size, cfg=cfg)
    task = get_loss(cfg.variant)(model, batch_for(cfg, tiny_tokenizer))
    total, extras = with_auxiliary_losses(model, task)
    assert total is task and extras == {}


def test_sampling_uses_the_test_time_loop_budget(tiny_tokenizer) -> None:
    cfg = shrink(load_config(REPO / "configs" / "looped" / "recurrent_depth.yaml"))
    cfg.model.loop.exit_tolerance = None  # run every requested pass
    model = get_variant(cfg.variant)(vocab_size=tiny_tokenizer.vocab_size, cfg=cfg)
    session = session_from_model(model, tiny_tokenizer, cfg, device="cpu")
    session.generate("hi", max_new_tokens=2)
    assert loop_stats(model)["loop/loops"] == cfg.sample.num_loops == 16
    session.generate("hi", max_new_tokens=2, num_loops=3)
    assert loop_stats(model)["loop/loops"] == 3
    assert model.backbone.stack.eval_loops is None  # the budget is scoped to the call


def test_loop_depth_metric(tiny_tokenizer) -> None:
    cfg = shrink(load_config(REPO / "configs" / "looped" / "autoregressive_act.yaml"))
    model = get_variant(cfg.variant)(vocab_size=tiny_tokenizer.vocab_size, cfg=cfg).eval()
    (metric,) = instantiate_metrics(["loop_depth"], cfg.variant)
    metric.reset()
    batch = batch_for(cfg, tiny_tokenizer)
    with torch.no_grad():
        for _ in range(2):
            loss = get_loss(cfg.variant)(model, batch)
            metric.update(MetricContext(model=model, batch=batch, loss=loss, variant=cfg.variant))
    out = metric.compute()
    assert set(out) == {"loops", "ponder_cost", "mean_updates"} and 1 <= out["loops"] <= 8

    plain = get_variant("autoregressive")(
        vocab_size=tiny_tokenizer.vocab_size,
        cfg=shrink(load_config(REPO / "configs" / "autoregressive" / "nano.yaml")),
    )
    metric.reset()
    metric.update(MetricContext(model=plain, batch=batch, loss=loss, variant="autoregressive"))
    assert metric.compute() == {}


def test_a_looped_checkpoint_loads_as_a_vlm_backbone(tiny_tokenizer, tmp_path, monkeypatch) -> None:
    """hale-autoregressive rebuilds any arch from the checkpoint's model section (VLM / VLA)."""
    pytest.importorskip("transformers")
    from hale_core.nn.llm import get_llm_backbone_spec
    from hale_core.nn.llm.hale_backbone import HaleSequenceLMBackbone

    cfg = shrink(load_config(REPO / "configs" / "looped" / "autoregressive_act.yaml"))
    model = get_variant(cfg.variant)(vocab_size=tiny_tokenizer.vocab_size, cfg=cfg).eval()
    path = tmp_path / "looped.pt"
    torch.save(
        {
            "variant": cfg.variant,
            "vocab_size": tiny_tokenizer.vocab_size,
            "config": cfg.model_dump(),
            "model_state_dict": model.state_dict(),
        },
        path,
    )
    monkeypatch.setattr(
        "transformers.AutoTokenizer.from_pretrained", lambda *_a, **_k: tiny_tokenizer
    )
    tiny_tokenizer.pad_token = "<pad>"
    llm = HaleSequenceLMBackbone.from_checkpoint(
        get_llm_backbone_spec("hale-autoregressive"), str(path)
    ).eval()
    ids = torch.randint(2, 60, (1, 8))
    with torch.no_grad():
        torch.testing.assert_close(llm(input_ids=ids).logits, model({"input_ids": ids}))
