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
    assert {"looped", "diffusion", "block_diffusion"} <= variants
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
    cfg = shrink(load_config(REPO / "configs" / "looped" / "act.yaml"))
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
    cfg = shrink(load_config(REPO / "configs" / "looped" / "act.yaml"))
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
    """hale-looped rebuilds a looped checkpoint as a Hale-VLM / Hale-VLA language backbone."""
    pytest.importorskip("transformers")
    from hale_core.nn.llm import get_llm_backbone_spec
    from hale_core.nn.llm.hale_backbone import HaleSequenceLMBackbone

    cfg = shrink(load_config(REPO / "configs" / "looped" / "act.yaml"))
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
        get_llm_backbone_spec("hale-looped"), str(path)
    ).eval()
    ids = torch.randint(2, 60, (1, 8))
    with torch.no_grad():
        torch.testing.assert_close(llm(input_ids=ids).logits, model({"input_ids": ids}))


# ---------------------------------------------------------------------------- variant: looped
def looped_lm(tiny_tokenizer, name: str = "nano"):
    cfg = shrink(load_config(REPO / "configs" / "looped" / f"{name}.yaml"))
    torch.manual_seed(0)
    return get_variant("looped")(vocab_size=tiny_tokenizer.vocab_size, cfg=cfg), cfg


def test_looped_variant_forces_the_looped_backbone_and_a_causal_mask(tiny_tokenizer) -> None:
    from hale_llm.models.models_looped import LoopedLM

    cfg = shrink(load_config(REPO / "configs" / "looped" / "nano.yaml"))
    cfg.model.arch = "default"  # the variant does not depend on the YAML remembering `arch`
    model = LoopedLM(tiny_tokenizer.vocab_size, cfg)
    assert isinstance(model.backbone, LoopedBackbone) and model.loop.n_loops == 4
    cfg.model.attn_type = "bidirectional"
    with pytest.raises(ValueError, match="causal"):
        LoopedLM(tiny_tokenizer.vocab_size, cfg)


def test_looped_variant_logits_are_causal_and_follow_the_loop_budget(tiny_tokenizer) -> None:
    model, cfg = looped_lm(tiny_tokenizer)
    model.eval()
    batch = batch_for(cfg, tiny_tokenizer)
    with torch.no_grad():
        base = model(batch)
        assert model.last_loop_stats.loops == 4
        longer = model.forward_with_loops(batch, 7)
        assert model.last_loop_stats.loops == 7 and not torch.allclose(base, longer)
        edited = {**batch, "input_ids": batch["input_ids"].clone()}
        edited["input_ids"][:, 10] = (edited["input_ids"][:, 10] + 1) % tiny_tokenizer.vocab_size
        torch.testing.assert_close(model(edited)[:, :10], base[:, :10])


def test_looped_sampler_records_the_passes_per_token(tiny_tokenizer) -> None:
    from hale_llm.models.models_looped.sample import sample_looped

    model, _ = looped_lm(tiny_tokenizer, "act")
    prompt = torch.randint(2, 60, (1, 4))
    out, history = sample_looped(
        model, tiny_tokenizer, prompt, max_new_tokens=3, return_history=True
    )
    assert out.shape == (1, 7) and len(history) == 4 and len(model.loops_per_token) == 3
    sample_looped(model, tiny_tokenizer, prompt, max_new_tokens=2, num_loops=2)
    assert all(n <= 2 for n in model.loops_per_token)
    assert model.backbone.stack.eval_loops is None


def test_loop_scaling_and_ponder_histogram_metrics(tiny_tokenizer) -> None:
    model, cfg = looped_lm(tiny_tokenizer, "act")
    model.eval()
    scaling, histogram = instantiate_metrics(["loop_scaling", "ponder_histogram"], "looped")
    batch = batch_for(cfg, tiny_tokenizer)
    for metric in (scaling, histogram):
        metric.reset()
        with torch.no_grad():
            metric.update(
                MetricContext(model=model, batch=batch, loss=torch.tensor(0.0), variant="looped")
            )
    assert set(scaling.compute()) == {"loss@T1", "loss@T4", "loss@T8", "loss@T16"}
    hist = histogram.compute()
    fracs = {k: v for k, v in hist.items() if k.startswith("ponder_frac@")}
    assert sum(fracs.values()) == pytest.approx(1.0)
    assert 1 <= hist["ponder_min"] <= hist["ponder_mean"] <= hist["ponder_max"] <= 8

    plain, _ = looped_lm(tiny_tokenizer)  # no ACT -> no histogram
    histogram.reset()
    histogram.update(
        MetricContext(model=plain, batch=batch, loss=torch.tensor(0.0), variant="looped")
    )
    assert histogram.compute() == {}


def test_looped_variant_has_the_token_metrics() -> None:
    names = [
        type(m).__name__
        for m in instantiate_metrics(
            ["loss", "token_accuracy", "topk_accuracy", "loop_depth", "loop_scaling"], "looped"
        )
    ]
    assert names == [
        "LossMetric",
        "TokenAccuracyMetric",
        "TopKAccuracyMetric",
        "LoopDepthMetric",
        "LoopScalingMetric",
    ]
