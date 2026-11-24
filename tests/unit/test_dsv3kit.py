"""Tests for notebooks/deepseek_v3/dsv3kit.py (from-scratch DeepSeek-V3 blocks)."""
import importlib.util, math, sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

KIT = Path(__file__).resolve().parents[2] / "notebooks" / "deepseek_v3" / "dsv3kit.py"
spec = importlib.util.spec_from_file_location("dsv3kit", KIT)
K = importlib.util.module_from_spec(spec)
sys.modules["dsv3kit"] = K
spec.loader.exec_module(K)
torch.manual_seed(0)


def test_rope_is_relative_and_norm_preserving():
    rope = K.Rope(16)
    q, k = torch.randn(16), torch.randn(16)
    def dot(m, n):
        qm = rope.apply(q.view(1, 1, 16).expand(1, m + 1, 16))[0, m]
        kn = rope.apply(k.view(1, 1, 16).expand(1, n + 1, 16))[0, n]
        return float(qm @ kn)
    assert dot(5, 2) == pytest.approx(dot(13, 10), abs=1e-4)          # same offset 3
    assert dot(5, 2) != pytest.approx(dot(5, 3), abs=1e-3)
    x = torch.randn(2, 7, 16)
    assert torch.allclose(rope.apply(x).norm(dim=-1), x.norm(dim=-1), atol=1e-5)


def test_yarn_limits():
    inv, gamma = K.yarn_inv_freq(64, scale=40, orig_T=4096)
    base = K.rope_inv_freq(64)
    assert gamma[0] == 1 and torch.allclose(inv[0], base[0])            # fastest channel unchanged
    assert gamma[-1] == 0 and torch.allclose(inv[-1], base[-1] / 40)    # slowest channel interpolated by s
    inv1, _ = K.yarn_inv_freq(64, scale=1, orig_T=4096)
    assert torch.allclose(inv1, base)
    assert K.yarn_mscale(40) == pytest.approx(0.1 * math.log(40) + 1)
    assert bool((inv <= base + 1e-9).all())


def make_mla():
    return K.MLA(d=64, n_h=4, d_h=16, d_c=12, d_cq=20, d_r=8)


def test_mla_absorbed_equals_expanded_and_cache_decode():
    m = make_mla().eval()
    h = torch.randn(2, 11, 64)
    full = m(h)
    ab, cache = m.forward_absorbed(h)
    assert torch.allclose(full, ab, atol=1e-5)
    # incremental: prefill 7 tokens then 4 single-token steps == full
    out, cache = m.forward_absorbed(h[:, :7])
    outs = [out]
    for t in range(7, 11):
        o, cache = m.forward_absorbed(h[:, t:t + 1], cache)
        outs.append(o)
    assert torch.allclose(torch.cat(outs, 1), full, atol=1e-5)
    assert cache[0].shape == (2, 11, 12) and cache[1].shape == (2, 11, 8)


def test_mla_causal():
    m = make_mla().eval()
    h = torch.randn(1, 9, 64)
    h2 = h.clone(); h2[:, 6:] = torch.randn(1, 3, 64)
    assert torch.allclose(m(h)[:, :6], m(h2)[:, :6], atol=1e-5)


def test_kv_cache_numbers_of_the_paper():
    assert K.kv_cache_per_token("mha") == 32768
    assert K.kv_cache_per_token("mla") == 576
    assert K.kv_cache_per_token("mha") / K.kv_cache_per_token("mla") == pytest.approx(56.9, abs=0.1)
    assert K.kv_cache_per_token("mla") == pytest.approx(4.5 * 128)       # = GQA with 2.25 groups


def dense_moe_reference(moe, x):
    u = x.reshape(-1, x.shape[-1])
    aff, idx, g = moe.route(u)
    gate = torch.zeros_like(aff).scatter(1, idx, g)                      # (N, Nr) dense gating matrix
    out = sum(sh(u) for sh in moe.shared)
    for e, ex in enumerate(moe.routed):
        out = out + gate[:, e:e + 1] * ex(u)
    return out.view_as(x), gate


def test_moe_matches_dense_reference_and_gates_sum_to_one():
    moe = K.DeepSeekMoE(32, 16, 1, 8, 3)
    x = torch.randn(2, 10, 32)
    y = moe(x)
    ref, gate = dense_moe_reference(moe, x)
    assert torch.allclose(y, ref, atol=1e-5)
    assert torch.allclose(gate.sum(-1), torch.ones(20), atol=1e-5)
    assert ((gate > 0).sum(-1) == 3).all()
    assert int(moe.last["load"].sum()) == 20 * 3


def test_bias_changes_routing_but_not_gate_values():
    moe = K.DeepSeekMoE(32, 16, 0, 8, 2)
    u = torch.randn(50, 32)
    aff0, idx0, _ = moe.route(u)
    moe.bias[:] = torch.tensor([5.0, 0, 0, 0, 0, 0, 0, -5.0])           # favour expert 0, avoid expert 7
    aff1, idx1, g1 = moe.route(u)
    assert torch.equal(aff0, aff1)                                       # affinities untouched
    assert (idx1 == 0).any(-1).all() and not (idx1 == 7).any()
    assert torch.allclose(g1, aff1.gather(1, idx1) / aff1.gather(1, idx1).sum(-1, keepdim=True))


def test_node_limited_routing():
    moe = K.DeepSeekMoE(32, 8, 0, 16, 4, n_groups=4, topk_groups=2)
    u = torch.randn(200, 32)
    _, idx, _ = moe.route(u)
    groups = idx // 4
    n_used = torch.tensor([len(set(r.tolist())) for r in groups])
    assert int(n_used.max()) <= 2


def test_update_bias_direction_and_seq_loss_value():
    moe = K.DeepSeekMoE(16, 8, 0, 4, 1)
    moe.last = dict(load=torch.tensor([10, 0, 5, 5]), aff=torch.rand(20, 4), idx=torch.zeros(20, 1, dtype=torch.long))
    K.update_bias(moe, 0.1)
    assert moe.bias[0] < 0 and moe.bias[1] > 0           # overloaded down, underloaded up
    # perfectly balanced routing and uniform affinities -> f_i = 1, P_i = 1/Nr  => loss = alpha
    B, T, Nr, Kt = 2, 8, 4, 1
    idx = torch.arange(B * T).remainder(Nr).view(-1, 1)
    moe.last = dict(load=torch.bincount(idx.view(-1), minlength=Nr), aff=torch.full((B * T, Nr), 0.5), idx=idx)
    assert float(K.seq_balance_loss(moe, B, T, 0.01)) == pytest.approx(0.01, abs=1e-6)
    # every token goes to expert 0 and the router likes expert 0: f_0 = Nr/K = 4, P_0 = 0.75  => loss = 3 * alpha
    aff = torch.full((B * T, Nr), 0.1); aff[:, 0] = 0.9
    moe.last = dict(load=torch.tensor([B * T, 0, 0, 0]), aff=aff, idx=torch.zeros(B * T, 1, dtype=torch.long))
    assert float(K.seq_balance_loss(moe, B, T, 0.01)) == pytest.approx(0.03, abs=1e-6)


def tiny_cfg(**kw):
    return K.LMConfig(vocab=30, d=32, n_layers=2, n_heads=2, d_ff=48, d_c=8, d_cq=12, d_r=4, moe_ff=16,
                      n_routed=4, top_k=2, **kw)


def test_mtp_alignment_and_shapes():
    cfg = tiny_cfg(mtp_depth=2)
    m = K.TinyLM(cfg).eval()
    seq = torch.randint(0, 30, (3, 17))
    outs = K.mtp_forward(m, seq)
    assert [o[0].shape[1] for o in outs] == [15, 14]                     # T=16: depth k has T-k positions
    # targets of depth k are tokens k+1 .. T  -> seq[:, k+1 : k+1+n]
    assert torch.equal(outs[0][1], seq[:, 2:17]) and torch.equal(outs[1][1], seq[:, 3:17])
    # manual depth-1 computation
    h = m.trunk(seq[:, :-1])
    x = m.mtp[0].proj(torch.cat([m.mtp[0].n_h(h[:, :15]), m.mtp[0].n_e(m.emb(seq[:, 1:16]))], -1))
    manual = m.head(m.mtp[0].block(x))
    assert torch.allclose(manual, outs[0][0], atol=1e-6)
    total, parts = K.lm_loss(m, seq)
    exp = parts["main"] + 0.3 * (parts["mtp1"] + parts["mtp2"]) / 2
    assert float(total) == pytest.approx(float(exp), rel=1e-5)


def test_mtp_is_causal():
    cfg = tiny_cfg(mtp_depth=1)
    m = K.TinyLM(cfg).eval()
    seq = torch.randint(0, 30, (1, 13))
    seq2 = seq.clone(); seq2[:, 8:] = torch.randint(0, 30, (1, 5))
    l1, l2 = K.mtp_forward(m, seq)[0][0], K.mtp_forward(m, seq2)[0][0]
    # MTP position i sees tokens <= i+1 ; positions 0..6 only see tokens 0..7
    assert torch.allclose(l1[:, :7], l2[:, :7], atol=1e-5)
    assert not torch.allclose(l1[:, 7:], l2[:, 7:], atol=1e-5)


def test_tinylm_all_variants_train_a_step():
    for kw in (dict(), dict(attn="mla"), dict(ffn="moe"), dict(attn="mla", ffn="moe", mtp_depth=1)):
        cfg = tiny_cfg(**kw)
        m = K.TinyLM(cfg)
        seq = torch.randint(0, 30, (2, 17))
        loss, _ = K.lm_loss(m, seq, seq_balance_alpha=0.001 if cfg.ffn == "moe" else 0.0)
        loss.backward()
        assert math.isfinite(float(loss)) and all(p.grad is not None for p in m.parameters() if p.requires_grad and p.numel() > 0 and "bias" not in "")


def test_charData():
    d = K.CharData()
    b = d.batch("train", 4, 32, torch.Generator().manual_seed(0))
    assert b.shape == (4, 33) and d.vocab > 50


def test_fp8_roundtrip_and_fine_grained_beats_per_tensor():
    x = torch.randn(64, 256)
    x[:, 5] *= 100000                                                      # an outlier channel
    q, s = K.quant_tile(x)
    e_tile = (K.dequant_tile(q, s) - x).norm() / x.norm()
    qt, st = K.quant_tensor(x)
    e_tensor = ((qt.float() * st) - x).norm() / x.norm()
    assert e_tile < 0.05 and e_tile < e_tensor
    # normal (non-outlier) channels are much better off with per-tile scales
    mask = torch.ones(256, dtype=torch.bool); mask[5] = False
    ok_tile = ((K.dequant_tile(q, s) - x)[:, mask]).norm() / x[:, mask].norm()
    ok_tensor = (((qt.float() * st) - x)[:, mask]).norm() / x[:, mask].norm()
    assert ok_tile < 0.5 * ok_tensor


def test_block_quant_roundtrip():
    w = torch.randn(256, 256)
    q, s = K.quant_block(w)
    assert s.shape == (2, 2)
    assert (K.dequant_block(q, s) - w).norm() / w.norm() < 0.04


def test_fp8_gemm_exact_accumulation_matches_dequantised_matmul():
    a, b = torch.randn(8, 256), torch.randn(16, 256)
    aq, sa = K.quant_tile(a); bq, sb = K.quant_tile(b)
    ref = K.dequant_tile(aq, sa) @ K.dequant_tile(bq, sb).t()
    got = K.fp8_gemm(aq, sa, bq, sb, acc_bits=None)
    assert torch.allclose(got, ref, rtol=1e-4, atol=1e-4)
    lim = K.fp8_gemm(aq, sa, bq, sb, acc_bits=14)
    assert (lim - ref).abs().max() / ref.abs().max() < 5e-3


def test_round_sig_bits():
    x = torch.tensor([1.0, 1.0 + 2 ** -10, 3.14159265])
    assert torch.equal(K.round_sig_bits(x, 24)[:2], x[:2])
    assert float(K.round_sig_bits(torch.tensor(1.0 + 2 ** -10), 8)) == 1.0


def test_fp8_linear_forward_backward_close_to_fp32():
    torch.manual_seed(1)
    lin = K.FP8Linear(256, 128)
    x = torch.randn(48, 256, requires_grad=True)
    y = lin(x)
    ref = x @ lin.weight.t()
    assert y.shape == (48, 128) and (y - ref).norm() / ref.norm() < 0.05
    dy = torch.randn_like(y)
    y.backward(dy)
    dx_ref, dw_ref = dy @ lin.weight.detach(), dy.t() @ x.detach()
    assert (x.grad - dx_ref).norm() / dx_ref.norm() < 0.06
    assert (lin.weight.grad - dw_ref).norm() / dw_ref.norm() < 0.06
    assert x.grad.shape == x.shape and lin.weight.grad.shape == lin.weight.shape
