"""Block flow matching (variant: block_flow_matching): path, model, loss, sampler, metric."""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch

import hale_llm.models  # noqa: F401
from hale_llm.config.schema import load_config
from hale_llm.core.registry import get_loss, get_sampler, get_variant, instantiate_metrics
from hale_llm.dataloader.collate import get_collate
from hale_llm.exceptions import ModelConfigurationError
from hale_llm.inference.session import session_from_model
from hale_llm.metrics.base import MetricContext
from hale_llm.models.models_block_flow_matching import MaskingPath, corrupt_blocks

REPO = Path(__file__).resolve().parents[2]
CONFIGS = sorted((REPO / "configs" / "block_flow_matching").glob("*.yaml"))


def tiny(name: str = "small", **model):
    cfg = load_config(REPO / "configs" / "block_flow_matching" / f"{name}.yaml")
    cfg.device = "cpu"
    cfg.model.d_model, cfg.model.n_heads, cfg.model.n_layers, cfg.model.d_ff = 32, 4, 1, 64
    cfg.model.max_length, cfg.model.block_size = 16, 4
    cfg.sample.sampling_steps = 4
    for key, value in model.items():
        setattr(cfg.model, key, value)
    return cfg


def build(cfg, tokenizer):
    torch.manual_seed(0)
    return get_variant("block_flow_matching")(vocab_size=tokenizer.vocab_size, cfg=cfg)


def batch_for(cfg, tokenizer) -> dict:
    ids = torch.randint(
        2, tokenizer.vocab_size, (2, 16), generator=torch.Generator().manual_seed(0)
    )
    ids[1, -3:] = tokenizer.pad_token_id
    return get_collate("block_flow_matching")(ids, tokenizer)


# ---------------------------------------------------------------------------- path
@pytest.mark.parametrize("schedule", ["linear", "cosine"])
def test_masking_path_schedule(schedule: str) -> None:
    path = MaskingPath(schedule)
    t = torch.linspace(0, 0.99, 50)
    kappa = path.kappa(t)
    assert float(path.kappa(torch.tensor(0.0))) == 0.0
    assert math.isclose(float(path.kappa(torch.tensor(1.0))), 1.0, abs_tol=1e-6)
    assert (kappa[1:] >= kappa[:-1]).all()
    # rate = kappa' / (1 - kappa), checked against a finite difference
    h = 1e-4
    numeric = (path.kappa(t + h) - path.kappa(t)) / h / (1 - kappa)
    torch.testing.assert_close(path.rate(t), numeric, rtol=2e-2, atol=2e-2)
    # Euler steps compose to the exact reveal probability of the path
    steps = torch.linspace(0, 1, 9).tolist()
    still_masked = 1.0
    for a, b in zip(steps[:-1], steps[1:], strict=True):
        still_masked *= 1 - path.unmask_probability(a, b)
    assert still_masked == pytest.approx(0.0, abs=1e-9)
    assert path.unmask_probability(0.0, 0.5) == pytest.approx(float(path.kappa(torch.tensor(0.5))))
    with pytest.raises(ValueError):
        MaskingPath("sigmoid")


def test_corruption_draws_one_time_per_block() -> None:
    torch.manual_seed(0)
    x1 = torch.randint(2, 50, (64, 32))
    x_t, is_masked, t = corrupt_blocks(x1, 8, 1, MaskingPath())
    blocks = t.view(64, 4, 8)
    assert torch.equal(blocks, blocks[..., :1].expand_as(blocks))
    assert torch.equal(x_t[is_masked], torch.ones_like(x_t[is_masked]))
    assert torch.equal(x_t[~is_masked], x1[~is_masked])
    # P(masked) = 1 - kappa(t): check on the empirical mean
    assert float(is_masked.float().mean()) == pytest.approx(float((1 - t).mean()), abs=0.03)
    with pytest.raises(ValueError, match="divisible"):
        corrupt_blocks(x1[:, :30], 8, 1, MaskingPath())


# ---------------------------------------------------------------------------- model / loss
@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.stem)
def test_configs_train_one_step(path: Path, tiny_tokenizer) -> None:
    cfg = tiny(path.stem)
    model = build(cfg, tiny_tokenizer)
    loss = get_loss("block_flow_matching")(model, batch_for(cfg, tiny_tokenizer))
    assert torch.isfinite(loss) and loss > 0
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.backbone.parameters())


def test_model_validates_its_layout(tiny_tokenizer) -> None:
    with pytest.raises(ValueError, match="block_causal"):
        build(tiny(attn_type="bidirectional"), tiny_tokenizer)
    with pytest.raises(ValueError, match="use_time_cond"):
        build(tiny(use_time_cond=False), tiny_tokenizer)
    with pytest.raises(ValueError, match="divisible"):
        build(tiny(block_size=5), tiny_tokenizer)


def test_noised_block_sees_only_earlier_clean_blocks(tiny_tokenizer) -> None:
    cfg = tiny()
    model = build(cfg, tiny_tokenizer).eval()
    x = torch.randint(2, 60, (1, 16))
    t = torch.full((1, 16), 0.3)
    with torch.no_grad():
        ref = model({"x_clean": x, "x_t": x, "t": t})
        later = x.clone()
        later[:, 8:] = (later[:, 8:] + 1) % 60  # blocks 2, 3 change (clean and noised)
        out = model({"x_clean": later, "x_t": later, "t": t})
    torch.testing.assert_close(out[:, :8], ref[:, :8])  # blocks 0, 1 untouched
    assert not torch.allclose(out[:, 8:], ref[:, 8:])


def test_loss_weights_follow_the_path_rate(tiny_tokenizer, monkeypatch) -> None:
    """With t fixed the loss is rate(t) x the masked-token NLL."""
    cfg = tiny()
    model = build(cfg, tiny_tokenizer)
    batch = batch_for(cfg, tiny_tokenizer)
    batch["attention_mask"] = torch.ones_like(batch["input_ids"])
    monkeypatch.setattr(
        MaskingPath, "sample_times", lambda self, b, nb, device: torch.full((b, nb), 0.5)
    )
    torch.manual_seed(1)
    loss = get_loss("block_flow_matching")(model, batch)
    torch.manual_seed(1)
    x_t, masked, t = corrupt_blocks(batch["input_ids"], 4, batch["mask_token_id"], model.path)
    logits = model({**batch, "x_clean": batch["input_ids"], "x_t": x_t, "t": t})
    nll = torch.nn.functional.cross_entropy(
        logits.transpose(1, 2), batch["input_ids"], reduction="none"
    )
    expected = 2.0 * nll[masked].mean()  # linear path: rate(0.5) = 1 / (1 - 0.5)
    assert cfg.train.loss_type == "ce"
    torch.testing.assert_close(loss, expected, rtol=1e-4, atol=1e-4)


# ---------------------------------------------------------------------------- sampling / eval
def test_sampler_fills_blocks_left_to_right_and_keeps_the_prompt(tiny_tokenizer) -> None:
    cfg = tiny()
    model = build(cfg, tiny_tokenizer)
    prompt = torch.randint(2, 60, (2, 6))
    out, history = get_sampler("block_flow_matching")(
        model, tiny_tokenizer, prompt, max_new_tokens=5, return_history=True
    )
    assert out.shape == (2, 12)  # prompt 6 + 5 new -> 11 tokens -> 3 blocks of 4
    assert torch.equal(out[:, :6], prompt)
    assert not out.eq(tiny_tokenizer.mask_token_id).any()
    blocks = [meta["block"] for _, _, meta in history[1:]]
    assert blocks == sorted(blocks) and set(blocks) == {1, 2}  # block 0 is all prompt
    times = [t for t, _, meta in history[1:] if meta["block"] == 2]
    assert times == sorted(times) and times[-1] == 1.0  # integrates t: 0 -> 1


def test_session_and_masked_accuracy(tiny_tokenizer) -> None:
    cfg = tiny()
    model = build(cfg, tiny_tokenizer)
    session = session_from_model(model, tiny_tokenizer, cfg, device="cpu")
    result = session.generate("hi", max_new_tokens=4, return_history=True)
    assert result.token_ids.size(1) == 8 and result.history
    (metric,) = instantiate_metrics(["masked_accuracy"], "block_flow_matching")
    metric.reset()
    batch = batch_for(cfg, tiny_tokenizer)
    with torch.no_grad():
        metric.update(
            MetricContext(
                model=model, batch=batch, loss=torch.tensor(0.0), variant="block_flow_matching"
            )
        )
    assert 0.0 <= metric.compute()["masked_accuracy"] <= 1.0


def test_kappa_schedules_are_registered_and_unknown_names_list_known_ones():
    from hale_llm.models.models_block_flow_matching.kappa_schedule import KAPPA_SCHEDULES

    assert {"linear", "cosine"} <= set(KAPPA_SCHEDULES)
    with pytest.raises(ModelConfigurationError, match="known: .*cosine"):
        MaskingPath("sigmoid")
