"""Selectable feed-forward activations (dense MLP and experts) and an independent check of the DeepSeek MoE."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

import hale_llm.models  # noqa: F401
from hale_llm.config.schema import load_config
from hale_llm.config.sections.model import ModelConfig
from hale_llm.nn.activation import ACTIVATIONS, get_activation, register_activation
from hale_llm.nn.blocks.feed_forward import FeedForward
from hale_llm.nn.moe import DeepseekMoE, Expert
from hale_llm.registry import get_variant
from hale_llm.tokenizer.vocab.param_estimate import estimate_non_embedding_params

# Written out by hand, not taken from the registry, so a wrong registration is caught.
PLAIN = {"gelu": lambda z: 0.5 * z * (1 + torch.erf(z / 2**0.5)), "relu": lambda z: z.clamp(min=0), "silu": lambda z: z * torch.sigmoid(z)}
GATED = {"swiglu": PLAIN["silu"], "geglu": PLAIN["gelu"], "reglu": PLAIN["relu"]}
ALL = [*PLAIN, *GATED]


def test_every_documented_activation_is_registered_with_the_right_kind():
    assert set(ALL) <= set(ACTIVATIONS)
    assert [get_activation(n).gated for n in PLAIN] == [False] * 3
    assert [get_activation(n).gated for n in GATED] == [True] * 3


def test_a_second_registration_of_a_name_is_refused_and_unknown_names_list_the_options():
    with pytest.raises(ValueError, match="already registered"):
        register_activation("gelu", F.gelu)
    with pytest.raises(KeyError, match="registered=.*gelu"):
        get_activation("nope")


# ----- dense feed-forward -----
@pytest.mark.parametrize("name", ALL)
def test_dense_feed_forward_equals_the_formula_written_out_from_its_weights(name):
    torch.manual_seed(0)
    ffn = FeedForward(8, 20, activation=name).eval()
    x = torch.randn(2, 5, 8)
    pre = x @ ffn.up.weight.T + ffn.up.bias
    if name in GATED:
        hidden = GATED[name](pre) * (x @ ffn.gate.weight.T + ffn.gate.bias)
    else:
        hidden = PLAIN[name](pre)
    expected = hidden @ ffn.down.weight.T + ffn.down.bias
    assert torch.allclose(ffn(x), expected, atol=1e-5)


@pytest.mark.parametrize("name", ALL)
def test_dense_feed_forward_parameter_count_and_shape(name):
    ffn = FeedForward(8, 20, activation=name)
    plain = 8 * 20 + 20 + 20 * 8 + 8  # up + bias, down + bias
    gate = 8 * 20 + 20
    assert sum(p.numel() for p in ffn.parameters()) == plain + (gate if name in GATED else 0)
    assert ffn(torch.randn(3, 4, 8)).shape == (3, 4, 8)


def test_the_default_is_gelu_and_has_exactly_the_old_parameters_in_the_old_order():
    ffn = FeedForward(8, 20)
    assert [n for n, _ in ffn.named_parameters()] == ["up.weight", "up.bias", "down.weight", "down.bias"]
    x = torch.randn(2, 3, 8)
    assert torch.allclose(ffn(x), torch.nn.functional.linear(F.gelu(F.linear(x, ffn.up.weight, ffn.up.bias)), ffn.down.weight, ffn.down.bias))
    assert ModelConfig().ffn_activation == "gelu" and ModelConfig().moe_activation == "swiglu"


@pytest.mark.parametrize("name", ALL)
def test_every_activation_trains_all_its_parameters_and_the_gate_matters(name):
    torch.manual_seed(1)
    ffn = FeedForward(8, 20, activation=name).train()
    ffn(torch.randn(2, 5, 8)).pow(2).sum().backward()
    for parameter_name, parameter in ffn.named_parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all() and parameter.grad.abs().sum() > 0, parameter_name


def test_different_activations_give_different_outputs():
    x = torch.randn(2, 5, 8)
    outputs = []
    for name in PLAIN:
        torch.manual_seed(2)
        outputs.append(FeedForward(8, 20, activation=name).eval()(x))
    assert all(not torch.allclose(outputs[i], outputs[j], atol=1e-4) for i in range(3) for j in range(i + 1, 3))


# ----- through the config, the model and the parameter estimate -----
@pytest.mark.parametrize("name", ALL)
def test_the_config_choice_reaches_the_dense_blocks_of_the_trunk(name):
    cfg = load_config("configs/autoregressive_moe/nano.yaml")
    cfg.set_context_window(16)
    cfg.model.ffn_activation, cfg.model.ffn_type = name, "mlp"
    model = get_variant("autoregressive")(vocab_size=40, cfg=cfg)
    blocks = list(model.backbone.stack.layers)  # main keeps the blocks in a DecoderTransformer
    assert all((block.ffn.gate is not None) == (name in GATED) for block in blocks)
    ids = torch.randint(1, 40, (2, 16))
    logits = model({"input_ids": ids, "attention_mask": torch.ones_like(ids)})
    assert logits.shape == (2, 16, 40) and torch.isfinite(logits).all()


def test_an_unknown_activation_fails_when_the_model_is_built_with_the_valid_names():
    cfg = load_config("configs/autoregressive_moe/nano.yaml")
    cfg.set_context_window(16)
    cfg.model.ffn_type, cfg.model.ffn_activation = "mlp", "mish"
    with pytest.raises(KeyError, match="activation 'mish'.*registered=.*gelu"):
        get_variant("autoregressive")(vocab_size=40, cfg=cfg)


@pytest.mark.parametrize("ffn_type,field", [("mlp", "ffn_activation"), ("moe", "moe_activation")])
@pytest.mark.parametrize("name", ALL)
def test_the_parameter_estimate_matches_the_built_model(ffn_type, field, name):
    cfg = load_config("configs/autoregressive_moe/nano.yaml")
    cfg.variant = "autoregressive" if ffn_type == "mlp" else "autoregressive_moe"
    cfg.set_context_window(16)
    cfg.model.ffn_type = ffn_type
    setattr(cfg.model, field, name)
    model = get_variant(cfg.variant)(vocab_size=40, cfg=cfg)
    blocks = sum(p.numel() for n, p in model.named_parameters() if ".layers." in n)
    ignored = sum(p.numel() for n, p in model.named_parameters() if ".layers." in n and (n.endswith(".bias") or "norm" in n))
    estimate = estimate_non_embedding_params(cfg.model)
    assert abs(blocks - ignored - estimate) <= 0.02 * estimate + 64 * cfg.model.n_layers * 4, (name, blocks - ignored, estimate)


# ----- experts and the DeepSeek MoE, checked against formulas written out from scratch -----
@pytest.mark.parametrize("name", ALL)
def test_expert_equals_the_formula_written_out_from_its_weights(name):
    torch.manual_seed(0)
    expert = Expert(8, 16, activation=name).eval()
    x = torch.randn(5, 8)
    pre = x @ expert.w1.weight.T
    hidden = GATED[name](pre) * (x @ expert.w3.weight.T) if name in GATED else PLAIN[name](pre)
    assert torch.allclose(expert(x), hidden @ expert.w2.weight.T, atol=1e-5)


def _moe(activation="swiglu", router="topk", n_routed=5, top_k=2, n_shared=2, **kw):
    torch.manual_seed(4)
    return DeepseekMoE(8, 12, n_routed, top_k, n_shared, router=router, activation=activation, **kw)


def _independent_moe(moe, x):
    """Per token: all shared experts + gate-softmax-over-the-top-k-weighted routed experts. Plain Python loops."""
    rows = []
    for token in x.reshape(-1, x.size(-1)):
        score = token @ moe.router.gate.weight.T + moe.router.gate.bias  # [E]
        top = sorted(range(len(score)), key=lambda e: -float(score[e].detach()))[: moe.router.top_k]
        weight = torch.softmax(torch.stack([score[e] for e in top]), dim=0)
        out = sum(e(token[None])[0] for e in moe.shared_experts) if len(moe.shared_experts) else torch.zeros_like(token)
        for w, e in zip(weight, top):
            out = out + w * moe.routed_experts[e](token[None])[0]
        rows.append(out)
    return torch.stack(rows).reshape(x.shape)


@pytest.mark.parametrize("name", ALL)
@pytest.mark.parametrize("top_k,n_shared", [(1, 0), (2, 1), (5, 2)])
def test_moe_output_equals_an_independent_per_token_computation(name, top_k, n_shared):
    moe = _moe(name, top_k=top_k, n_shared=n_shared).eval()
    x = torch.randn(3, 7, 8)
    assert torch.allclose(moe(x), _independent_moe(moe, x), atol=1e-5)


def test_moe_balance_loss_equals_the_formula_counted_by_hand():
    moe = _moe(top_k=2).train()
    x = torch.randn(4, 9, 8)
    moe(x)
    flat = x.reshape(-1, 8)
    scores = flat @ moe.router.gate.weight.T + moe.router.gate.bias  # [T, E]
    probs = torch.softmax(scores, dim=-1)
    counts = torch.zeros(5)
    for row in scores:
        for e in row.topk(2).indices.tolist():
            counts[e] += 1
    fraction = counts / counts.sum()  # share of the routing decisions each expert received
    expected = 5 * float((fraction * probs.mean(0)).sum())
    assert float(moe.aux_loss) == pytest.approx(expected, rel=1e-5)
    assert torch.allclose(moe.expert_share, fraction, atol=1e-6)


def test_balance_loss_for_fully_collapsed_routing_and_for_spread_routing():
    moe = _moe(top_k=1, n_routed=4, n_shared=0).train()
    torch.nn.init.zeros_(moe.router.gate.weight)
    moe.router.gate.bias.data = torch.zeros(4)
    # Inputs are irrelevant with a zero gate, but the top-1 choice is then always expert 0: collapsed.
    moe(torch.randn(1, 32, 8))
    collapsed = float(moe.aux_loss)
    assert collapsed == pytest.approx(4 * 1.0 * 0.25)  # share (1,0,0,0) . uniform probs (.25) times E = 1.0
    # Strong random gates spread 4000 tokens over the experts: the loss falls to about 1, the balanced value.
    moe.router.gate.weight.data = torch.randn(4, 8) * 10
    x = torch.randn(1, 4000, 8)
    moe(x)
    assert 0.99 < float(moe.aux_loss) < 1.3
    assert float(moe.expert_share.sum()) == pytest.approx(1.0, abs=1e-6)


def test_balance_loss_gradient_flows_through_the_router_probabilities_only():
    moe = _moe().train()
    moe(torch.randn(2, 16, 8))
    moe.aux_loss.backward()
    assert moe.router.gate.weight.grad.abs().sum() > 0  # via the mean probabilities
    assert all(p.grad is None for e in moe.routed_experts for p in e.parameters())  # experts are untouched


def test_experts_nobody_chose_get_no_gradient_and_nothing_breaks():
    moe = _moe(top_k=1, n_routed=4, n_shared=0).train()
    torch.nn.init.zeros_(moe.router.gate.weight)
    moe.router.gate.bias.data = torch.tensor([5.0, 0.0, 0.0, 0.0])  # everyone picks expert 0
    out = moe(torch.randn(2, 6, 8))
    out.sum().backward()
    assert moe.routed_experts[0].w1.weight.grad.abs().sum() > 0
    assert all(moe.routed_experts[e].w1.weight.grad is None for e in (1, 2, 3))
    assert torch.isfinite(out).all()


def test_moe_is_deterministic_in_eval_even_with_the_noisy_router_and_batch_independent():
    moe = _moe(router="noisy_topk").eval()
    x = torch.randn(3, 6, 8)
    assert torch.equal(moe(x), moe(x))
    assert torch.allclose(moe(x)[1], moe(x[1:2])[0], atol=1e-6)  # a token's output does not depend on its batch
