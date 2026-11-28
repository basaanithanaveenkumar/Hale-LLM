"""Adding MTP must not change what the trunk learns from the main loss: the data and loss flow, compared across variants."""

from __future__ import annotations

import pytest
import torch

import hale_llm.models  # noqa: F401
from hale_llm.config.schema import load_config
from hale_llm.registry import get_loss, get_variant

PAIRS = [("autoregressive", "autoregressive_mtp", "mlp"), ("autoregressive_moe", "autoregressive_moe_mtp", "moe")]


def _model(variant, ffn_type, weight=None, balance=None):
    cfg = load_config("configs/autoregressive_mtp/dense_nano.yaml")
    cfg.variant = variant
    cfg.set_context_window(16)
    cfg.model.d_model, cfg.model.n_heads, cfg.model.n_layers, cfg.model.d_ff = 32, 4, 2, 64
    cfg.model.dropout, cfg.model.ffn_type, cfg.model.moe_hidden_dim, cfg.model.moe_router = 0.0, ffn_type, 48, "topk"
    if weight is not None:
        cfg.model.mtp_loss_weight, cfg.model.mtp_loss_weight_final = weight, None
    if balance is not None:
        cfg.model.moe_aux_loss_coef = balance
    torch.manual_seed(0)  # the trunk is built first, so the same seed gives the same trunk weights
    return get_variant(variant)(vocab_size=50, cfg=cfg)


def _batch(seed=1):
    ids = torch.randint(0, 50, (4, 16), generator=torch.Generator().manual_seed(seed))
    labels = ids.clone()
    labels[:, :-1], labels[:, -1] = ids[:, 1:], -100
    return {"input_ids": ids, "attention_mask": torch.ones_like(ids), "labels": labels, "pad_token_id": None}


@pytest.mark.parametrize("plain,mtp,ffn", PAIRS)
def test_the_mtp_model_has_exactly_the_same_trunk_and_the_same_next_token_loss(plain, mtp, ffn):
    base, extended = _model(plain, ffn).eval(), _model(mtp, ffn).eval()
    batch = _batch()
    assert torch.equal(base(batch), extended(batch))  # same logits: MTP sits beside the trunk, not inside it
    assert torch.equal(get_loss(plain)(base, batch), get_loss(mtp)(extended, batch))  # evaluation scores the next-token loss only


@pytest.mark.parametrize("plain,mtp,ffn", PAIRS)
def test_with_lambda_zero_the_trunk_gets_exactly_the_gradients_of_the_plain_model(plain, mtp, ffn):
    # The balancing coefficient is 0 here: see the next test for why it matters for mixture-of-experts MTP blocks.
    base, extended = _model(plain, ffn, balance=0.0).train(), _model(mtp, ffn, weight=0.0, balance=0.0).train()
    batch = _batch()
    get_loss(plain)(base, batch).backward()
    get_loss(mtp)(extended, batch).backward()
    for name, parameter in base.named_parameters():
        assert torch.equal(parameter.grad, dict(extended.named_parameters())[name].grad), name


def test_mixture_of_experts_mtp_blocks_balance_their_experts_and_that_gradient_reaches_the_trunk():
    """By design every expert layer, in the trunk and in the MTP blocks, gets the balancing loss (coefficient
    ``moe_aux_loss_coef``). The MTP blocks' routers read the trunk's hidden states, so even at lambda = 0 the trunk
    receives that (small) gradient. With the coefficient at 0 it is exactly the plain model's (test above)."""
    base, extended = _model("autoregressive_moe", "moe").train(), _model("autoregressive_moe_mtp", "moe", weight=0.0).train()
    batch = _batch()
    get_loss("autoregressive_moe")(base, batch).backward()
    get_loss("autoregressive_moe_mtp")(extended, batch).backward()
    differs = [n for n, p in base.named_parameters() if not torch.equal(p.grad, dict(extended.named_parameters())[n].grad)]
    assert differs and all(n.startswith("backbone.") for n in differs)  # only the shared trunk is affected, via the balancing term


@pytest.mark.parametrize("plain,mtp,ffn", PAIRS)
def test_with_lambda_above_zero_the_trunk_does_get_a_gradient_from_the_mtp_loss(plain, mtp, ffn):
    base, extended = _model(plain, ffn).train(), _model(mtp, ffn, weight=0.7).train()
    batch = _batch()
    get_loss(plain)(base, batch).backward()
    get_loss(mtp)(extended, batch).backward()
    changed = [name for name, p in base.named_parameters() if not torch.equal(p.grad, dict(extended.named_parameters())[name].grad)]
    assert any(name.startswith("backbone.layers.") for name in changed)  # the MTP loss reaches the shared trunk and embedding
    assert "backbone.token_emb.weight" in changed
