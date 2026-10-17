"""Multi-token prediction: per-head offsets, loss, self-speculative decoding, head metric."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

import hale_llm.models  # noqa: F401
from hale_llm.config.schema import load_config
from hale_llm.core.registry import get_loss, get_variant, instantiate_metrics
from hale_llm.dataloader.collate import get_collate
from hale_llm.metrics.base import MetricContext
from hale_llm.models.models_mtp import head_losses, sample_mtp

REPO = Path(__file__).resolve().parents[2]


def mtp(tiny_tokenizer, heads: int = 3):
    cfg = load_config(REPO / "configs" / "mtp" / "small.yaml")
    cfg.device = "cpu"
    cfg.model.d_model, cfg.model.n_heads, cfg.model.n_layers, cfg.model.d_ff = 32, 4, 1, 64
    cfg.model.max_length, cfg.model.n_mtp_heads, cfg.model.dropout = 24, heads, 0.0
    torch.manual_seed(0)
    return get_variant("mtp")(vocab_size=tiny_tokenizer.vocab_size, cfg=cfg), cfg


def batch(tiny_tokenizer) -> dict:
    ids = torch.randint(2, 60, (2, 12), generator=torch.Generator().manual_seed(0))
    ids[1, -2:] = 0
    return get_collate("mtp")(ids, tiny_tokenizer)


def test_heads_share_the_trunk_and_the_unembedding(tiny_tokenizer) -> None:
    model, _ = mtp(tiny_tokenizer)
    logits = model(batch(tiny_tokenizer))
    assert len(logits) == 3 and all(x.shape == (2, 12, 64) for x in logits)
    torch.testing.assert_close(
        logits[0],
        model.backbone(
            batch(tiny_tokenizer)["input_ids"],
            attention_mask=batch(tiny_tokenizer)["attention_mask"],
        ),
    )
    # extra heads start as the identity on the trunk -> same logits as head 0 at init
    torch.testing.assert_close(
        logits[1],
        model.backbone.lm_head(
            model.extra_heads[0].norm_out(
                model.trunk(
                    batch(tiny_tokenizer)["input_ids"], batch(tiny_tokenizer)["attention_mask"]
                )
            )
        ),
    )


def test_loss_scores_head_k_against_the_token_k_plus_one_ahead(tiny_tokenizer) -> None:
    model, _ = mtp(tiny_tokenizer)
    b = batch(tiny_tokenizer)
    losses = head_losses(model, b)
    logits = model(b)
    for k, loss in enumerate(losses):
        target = b["labels"][:, k + 1 :]
        nll = torch.nn.functional.cross_entropy(
            logits[k][:, : -(k + 1)].reshape(-1, 64),
            target.reshape(-1),
            ignore_index=0,
            label_smoothing=0.0,
        )
        assert model.cfg.train.loss_type == "ce"
        torch.testing.assert_close(loss, nll, rtol=1e-4, atol=1e-4)
    total = get_loss("mtp")(model, b)
    torch.testing.assert_close(total, torch.stack(losses).mean())
    total.backward()
    assert all(p.grad is not None for p in model.extra_heads.parameters())


@pytest.mark.parametrize("heads", [1, 2, 4])
def test_greedy_speculative_decoding_equals_greedy_autoregressive(
    tiny_tokenizer, heads: int
) -> None:
    model, _ = mtp(tiny_tokenizer, heads)
    for head in model.extra_heads:  # make the drafts imperfect so rejections happen
        torch.nn.init.normal_(head.mlp[-1].weight, std=0.5)
    prompt = torch.randint(2, 60, (1, 4), generator=torch.Generator().manual_seed(3))
    out = sample_mtp(model, tiny_tokenizer, prompt, max_new_tokens=30, temperature=0.0)
    x = prompt
    with torch.no_grad():
        for _ in range(30):
            window = x[:, -model.backbone.max_length :]
            nxt = model({"input_ids": window})[0][:, -1].argmax(-1, keepdim=True)
            x = torch.cat([x, nxt], dim=1)
    assert torch.equal(out, x)
    trace = model.speculative_trace
    assert sum(len(s.emitted) for s in trace) == 30
    assert 1.0 <= model.tokens_per_forward <= heads
    assert all(s.accepted <= len(s.drafts) <= heads - 1 for s in trace)


def test_speculative_history_and_sampling(tiny_tokenizer) -> None:
    model, _ = mtp(tiny_tokenizer)
    prompt = torch.randint(2, 60, (1, 3))
    out, history = sample_mtp(model, tiny_tokenizer, prompt, max_new_tokens=6, return_history=True)
    assert out.shape == (1, 9) and history[-1][1].shape == (1, 9)
    with pytest.raises(ValueError, match="one sequence"):
        sample_mtp(model, tiny_tokenizer, torch.ones(2, 3, dtype=torch.long), max_new_tokens=2)


def test_head_accuracy_metric(tiny_tokenizer) -> None:
    model, _ = mtp(tiny_tokenizer)
    (metric,) = instantiate_metrics(["head_accuracy"], "mtp")
    metric.reset()
    with torch.no_grad():
        metric.update(
            MetricContext(
                model=model, batch=batch(tiny_tokenizer), loss=torch.tensor(0.0), variant="mtp"
            )
        )
    out = metric.compute()
    assert set(out) == {"head0_accuracy", "head1_accuracy", "head2_accuracy"}
    assert all(0.0 <= v <= 1.0 for v in out.values())
