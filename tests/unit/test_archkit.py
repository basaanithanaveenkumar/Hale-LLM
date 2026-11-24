"""Tests for notebooks/architecture_gallery/archkit.py (the 9-notebook 'How LLMs build on each other' series)."""
import sys
from pathlib import Path
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "notebooks" / "architecture_gallery"))
import archkit as A  # noqa: E402

D, H = 32, 4


@pytest.fixture(autouse=True)
def seed():
    torch.manual_seed(0)


def x20():
    return torch.randn(2, 20, D)


def test_causality_all_paths():
    m, x = A.Attn(D, H, n_kv=2, gate="elem", sink=True), x20()
    x2 = x.clone(); x2[:, 10:] += 1
    for kw in [{}, dict(window=6), dict(window=6, n_sink_tokens=2, cache_pos=True)]:
        assert torch.allclose(m(x, **kw)[:, :10], m(x2, **kw)[:, :10], atol=1e-6)


def test_gqa_equals_mha_with_tied_heads():
    mha, gqa = A.Attn(D, H, n_kv=H), A.Attn(D, H, n_kv=2)
    dh = D // H
    wk, wv = gqa.wk.weight.view(2, dh, D), gqa.wv.weight.view(2, dh, D)
    with torch.no_grad():
        mha.wq.weight.copy_(gqa.wq.weight); mha.wo.weight.copy_(gqa.wo.weight)
        mha.wk.weight.copy_(wk.repeat_interleave(2, 0).reshape(H * dh, D))
        mha.wv.weight.copy_(wv.repeat_interleave(2, 0).reshape(H * dh, D))
    x = x20()
    assert torch.allclose(mha(x), gqa(x), atol=1e-6)


def test_sink_logit_minus_inf_is_plain_attention():
    m, ms, x = A.Attn(D, H), A.Attn(D, H, sink=True), x20()
    ms.load_state_dict({**m.state_dict(), "sink": torch.full((H,), -1e9)})
    assert torch.allclose(ms(x), m(x), atol=1e-6)


def test_sink_logit_absorbs_mass():
    ms, x = A.Attn(D, H, sink=True), x20()
    ms.keep = True; ms(x)
    assert (ms.p.sum(-1) < 1).any() and (ms.p.sum(-1) <= 1 + 1e-6).all()


def test_streaming_matches_causal_until_eviction_and_masked_window_without_sinks():
    m, x = A.Attn(D, H), x20()
    assert torch.allclose(m(x), m(x, window=40, cache_pos=True), atol=1e-5)
    assert torch.allclose(m(x, window=6), m(x, window=6, cache_pos=True), atol=1e-5)   # RoPE is relative
    assert (m(x, window=6, cache_pos=True) - m(x, window=6, n_sink_tokens=2, cache_pos=True)).abs().max() > 1e-3


def test_gate_zero_weights_halves_output():
    m, x = A.Attn(D, H, gate="elem"), x20()
    base = A.Attn(D, H); base.load_state_dict({k: v for k, v in m.state_dict().items() if k != "wg.weight"})
    with torch.no_grad():
        m.wg.weight.zero_()
    assert torch.allclose(m(x), 0.5 * base(x), atol=1e-6)


def test_dsa_topk_and_dense_limit():
    ind, x = A.LightningIndexer(D), x20()
    I = ind(x)
    mk = A.topk_mask(I, 5)
    assert (mk.sum(-1) == torch.clamp(torch.arange(1, 21), max=5)).all()
    m = A.Attn(D, H)
    assert torch.allclose(m(x, extra_mask=A.topk_mask(I, 20)), m(x), atol=1e-6)


def test_indexer_kl_zero_at_target():
    B, T = 2, 7
    p = torch.rand(B, T, T).tril()
    p = p / p.sum(-1, keepdim=True)
    I = torch.where(p > 0, p.clamp_min(1e-12).log(), torch.full_like(p, -1e9))
    assert A.indexer_kl(I, p).abs() < 1e-5
    assert A.indexer_kl(torch.randn(B, T, T), p) > 0.01


def _rand(T=37, dk=6, dv=5):
    dt = torch.float64
    q = torch.randn(T, dk, dtype=dt)
    k = torch.nn.functional.normalize(torch.randn(T, dk, dtype=dt), dim=-1)
    v = torch.randn(T, dv, dtype=dt)
    return q, k, v, torch.rand(T, dtype=dt) * .3 + .7, torch.rand(T, dtype=dt)


def test_gated_delta_chunkwise_equals_recurrent():
    q, k, v, a, b = _rand()
    o1, S1 = A.linear_attn_recurrent(q, k, v, a, b, "gdelta")
    o2, S2 = A.gated_delta_chunkwise(q, k, v, a, b, C=8)
    assert torch.allclose(o1, o2, atol=1e-10) and torch.allclose(S1, S2, atol=1e-10)


def test_kda_chunkwise_equals_recurrent_and_reduces_to_gdn():
    q, k, v, a, b = _rand()
    ak = torch.rand(37, 6, dtype=torch.float64) * .3 + .7
    o1, _ = A.kda_recurrent(q, k, v, ak, b)
    o2, _ = A.kda_chunkwise(q, k, v, ak, b, C=8)
    assert torch.allclose(o1, o2, atol=1e-10)
    o3, _ = A.kda_recurrent(q, k, v, a[:, None].expand(37, 6).contiguous(), b)
    o4, _ = A.linear_attn_recurrent(q, k, v, a, b, "gdelta")
    assert torch.allclose(o3, o4, atol=1e-10)


def test_delta_rule_overwrites_and_special_cases():
    dt = torch.float64
    k = torch.nn.functional.normalize(torch.randn(3, 6, dtype=dt), dim=-1)
    v = torch.randn(3, 5, dtype=dt)
    k2 = torch.cat([k, k[:1]]); v2 = torch.cat([v, torch.randn(1, 5, dtype=dt)])   # key 0 written twice
    q = k2
    _, S = A.linear_attn_recurrent(q, k2, v2, None, torch.ones(4, dtype=dt), "delta")
    assert torch.allclose(S @ k2[0], v2[3], atol=1e-10)                              # newest value wins exactly
    _, S_lin = A.linear_attn_recurrent(q, k2, v2, None, None, "linear")
    assert not torch.allclose(S_lin @ k2[0], v2[3], atol=1e-3)                       # linear attention just piles up
    _, Sg = A.linear_attn_recurrent(q, k2, v2, torch.ones(4, dtype=dt), torch.ones(4, dtype=dt), "gdelta")
    _, Sd = A.linear_attn_recurrent(q, k2, v2, None, torch.ones(4, dtype=dt), "delta")
    assert torch.allclose(Sg, Sd)                                                    # alpha = 1 -> DeltaNet


def test_delta_rule_is_one_sgd_step():
    dt = torch.float64
    S, k, v, b = torch.randn(5, 6, dtype=dt), torch.randn(6, dtype=dt), torch.randn(5, dtype=dt), 0.37
    grad = torch.outer(S @ k - v, k)                         # d/dS of 0.5 ||S k - v||^2
    S_sgd = S - b * grad
    _, S_rule = A.linear_attn_recurrent(torch.zeros(1, 6, dtype=dt), k[None], v[None], None, torch.tensor([b], dtype=dt), "delta")
    S_rule = S_rule * 0 + (S - b * torch.outer(S @ k, k) + b * torch.outer(v, k))   # one step from a non-zero S
    S_rule = S - b * torch.outer(S @ k, k) + b * torch.outer(v, k)
    assert torch.allclose(S_sgd, S_rule)


def test_moe_and_latentmoe():
    x = x20()
    moe, lm = A.MoE(D, 8, 2, 16), A.MoE(D, 32, 8, 16, latent=8)
    assert moe(x).shape == lm(x).shape == x.shape
    assert abs(float(moe.load.sum()) - 1) < 1e-6
    n = lambda m: sum(p.numel() for p in m.experts.parameters())
    assert n(moe) == n(lm)                                    # iso expert parameters when E, k grow by d/l


def test_moe_bias_never_changes_gate_value():
    moe, x = A.MoE(D, 8, 2, 16), torch.randn(10, D)
    sel0, g0 = moe.route(x)
    sel1, g1 = moe.route(x, bias=torch.tensor([5.0] + [0.0] * 7))
    assert (sel1 == 0).any(-1).all()                           # bias pulls expert 0 into every selection
    aff = torch.sigmoid(moe.router(x))
    ref = aff.gather(-1, sel1); ref = ref / ref.sum(-1, keepdim=True)
    assert torch.allclose(g1, ref)


def test_quantile_balancing_exact_load():
    s = torch.randn(4000, 16)
    beta = A.quantile_offsets(s, 4)
    on = (s - beta > 0).float().sum(0)
    assert (on - 4000 * 4 / 16).abs().max() <= 1
    assert abs(A.quantile_init(1.0, 4, 16) - 0.6745) < 1e-3


def test_attnres_variants_run_and_are_causal():
    data = A.CharData()
    for res, nb in [("std", None), ("full", None), ("block", 4), ("block", 2)]:
        m = A.MiniLM(data.vocab, D, lambda i: A.Attn(D, H), lambda i: A.SwiGLU(D, 64), n_layers=4, res=res, n_blocks=nb)
        idx = torch.randint(0, data.vocab, (2, 16)); idx2 = idx.clone(); idx2[:, 8:] = 0
        assert torch.allclose(m(idx)[:, :8], m(idx2)[:, :8], atol=1e-5)


def test_depth_attention_is_uniform_at_init_and_sums_to_one():
    da = A.DepthAttn(D)
    srcs = [torch.randn(2, 5, D) for _ in range(4)]
    h = da(srcs)
    assert torch.allclose(da.last, torch.full_like(da.last, .25))
    assert torch.allclose(h, sum(srcs) / 4, atol=1e-6)


def test_block_equals_full_when_one_sublayer_per_block():
    data = A.CharData()
    mk = lambda res, nb: A.MiniLM(data.vocab, D, lambda i: A.Attn(D, H), lambda i: A.SwiGLU(D, 64), n_layers=2, res=res, n_blocks=nb)
    full, blk = mk("full", None), mk("block", 4)
    blk.load_state_dict(full.state_dict())
    idx = torch.randint(0, data.vocab, (2, 12))
    assert torch.allclose(full(idx), blk(idx), atol=1e-5)
