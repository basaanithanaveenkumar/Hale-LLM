"""Routers, experts and the DeepSeek-style mixture of experts."""

import pytest
import torch

from hale_llm.nn.moe import ROUTERS, BestKRouter, DeepseekMoE, Expert, NoisyBestKRouter


def test_routers_are_registered():
    assert {"topk", "noisy_topk"} <= set(ROUTERS)


def test_router_weights_sum_to_one_over_exactly_k_experts():
    router = BestKRouter(d_model=8, n_experts=6, top_k=2)
    out = router(torch.randn(10, 8))
    assert torch.allclose(out.weights.sum(-1), torch.ones(10), atol=1e-6)
    assert ((out.weights > 0).sum(-1) == 2).all()
    assert out.indices.shape == (10, 2)
    assert torch.allclose(out.probs.sum(-1), torch.ones(10), atol=1e-6)


def test_router_rejects_bad_top_k():
    with pytest.raises(ValueError):
        BestKRouter(8, 4, top_k=5)


def test_noise_only_during_training():
    router = NoisyBestKRouter(8, 6, 2)
    x = torch.randn(20, 8)
    router.eval()
    assert torch.equal(router(x).indices, router(x).indices)
    router.train()
    torch.manual_seed(0)
    first = router(x).weights
    second = router(x).weights
    assert not torch.equal(first, second)


@pytest.mark.parametrize("activation, n_matrices", [("swiglu", 3), ("gelu", 2)])
def test_expert_parameter_count(activation, n_matrices):
    expert = Expert(8, 16, activation=activation)
    assert sum(p.numel() for p in expert.parameters()) == n_matrices * 8 * 16
    assert expert(torch.randn(3, 8)).shape == (3, 8)


def test_unknown_activation_is_rejected():
    with pytest.raises(ValueError):
        Expert(8, 16, activation="relu")


def _moe(**kwargs):
    defaults = dict(d_model=8, hidden_dim=16, n_routed_experts=4, top_k=2, n_shared_experts=1)
    return DeepseekMoE(**{**defaults, **kwargs})


def test_output_matches_a_per_token_reference():
    torch.manual_seed(0)
    moe = _moe(router="topk").eval()
    x = torch.randn(2, 3, 8)
    out = moe(x)
    flat = x.reshape(-1, 8)
    routing = moe.router(flat)
    expected = sum(e(flat) for e in moe.shared_experts)
    for token in range(flat.size(0)):
        for expert_id in routing.indices[token].tolist():
            expected[token] = expected[token] + routing.weights[token, expert_id] * moe.routed_experts[expert_id](flat[token])
    assert torch.allclose(out.reshape(-1, 8), expected, atol=1e-5)


def test_works_without_shared_experts():
    moe = _moe(n_shared_experts=0).eval()
    assert moe(torch.randn(1, 4, 8)).shape == (1, 4, 8)


def test_gradient_reaches_router_and_experts():
    moe = _moe().train()
    moe(torch.randn(2, 5, 8)).sum().backward()
    assert moe.router.gate.weight.grad is not None
    assert any(e.w1.weight.grad is not None for e in moe.routed_experts)


def test_load_balance_loss_is_at_least_one_and_only_computed_in_training():
    moe = _moe(router="topk").train()
    # A zero gate gives uniform probabilities; the loss is then n_experts * sum(fraction / n) >= 1.
    torch.nn.init.zeros_(moe.router.gate.weight)
    torch.nn.init.zeros_(moe.router.gate.bias)
    moe(torch.randn(1, 64, 8))
    assert float(moe.aux_loss.detach()) >= 1.0 - 1e-6
    moe.eval()
    moe(torch.randn(1, 4, 8))
    assert float(moe.aux_loss) == 0.0  # only computed while training


def test_bfloat16_autocast_keeps_working():
    moe = _moe().train()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        out = moe(torch.randn(2, 4, 8))
    assert torch.isfinite(out.float()).all()
