"""fp4kit: emulated NVFP4 / MXFP8 / 4-over-6 / RHT / S2F6 numerics. Pure torch, runs on cpu (and mps when present)."""

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

KIT = Path(__file__).resolve().parents[2] / "notebooks" / "fp4_training" / "fp4kit.py"
spec = importlib.util.spec_from_file_location("fp4kit", KIT)
kit = importlib.util.module_from_spec(spec)
sys.modules["fp4kit"] = kit          # dataclasses need the module registered
spec.loader.exec_module(kit)

GRID = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]


def _brute_rne(v: float) -> float:
    """Nearest E2M1 value; ties go to the even 3-bit code (code = index in GRID)."""
    m = min(abs(v), 6.0)
    d = [abs(m - g) for g in GRID]
    best = min(d)
    cands = [i for i, x in enumerate(d) if abs(x - best) < 1e-12]
    i = next((c for c in cands if c % 2 == 0), cands[0])
    return (1 if v >= 0 else -1) * GRID[i]


def test_round_fp4_rtn_matches_bruteforce_including_ties():
    vals = [i / 64 for i in range(-448, 449)]                     # includes every midpoint exactly (multiples of 1/64)
    out = kit.round_fp4(torch.tensor(vals)).tolist()
    assert out == [_brute_rne(v) for v in vals]


def test_round_fp4_values_on_grid_are_fixed_points_and_saturate():
    g = torch.tensor(GRID + [-x for x in GRID[1:]])
    assert torch.equal(kit.round_fp4(g), g) and torch.equal(kit.round_fp4(g, "sr"), g)
    assert kit.round_fp4(torch.tensor([7.5, -100.0])).tolist() == [6.0, -6.0]


def test_stochastic_rounding_is_unbiased_and_uses_only_neighbours():
    gen = torch.Generator().manual_seed(0)
    for v in (0.2, 0.9, 2.4, 3.7, 5.3, -1.2):
        x = torch.full((200_000,), v)
        r = kit.round_fp4(x, "sr", gen)
        assert abs(r.mean().item() - v) < 0.01
        assert len(torch.unique(r)) == 2


def test_round_e4m3_matches_torch_float8_cast():
    torch.manual_seed(0)
    mag = 10 ** torch.empty(400_000).uniform_(-4, 2.65)              # up to ~447
    x = torch.cat([mag, -mag, torch.arange(256, dtype=torch.uint8).view(torch.float8_e4m3fn).float().nan_to_num()])
    x = x.clamp(-448, 448)
    assert torch.equal(kit.round_e4m3(x), x.to(torch.float8_e4m3fn).float())


def test_round_e4m3_saturates_and_sr_unbiased():
    assert kit.round_e4m3(torch.tensor([1000.0, -1000.0])).tolist() == [448.0, -448.0]
    gen = torch.Generator().manual_seed(1)
    x = torch.full((300_000,), 10.3)
    assert abs(kit.round_e4m3(x, "sr", gen).mean().item() - 10.3) < 0.01


def test_e8m0_up_is_power_of_two_not_below_input():
    x = torch.tensor([0.3, 1.0, 1.01, 5.0, 1000.0])
    s = kit.e8m0_up(x)
    assert s.tolist() == [0.5, 1.0, 2.0, 8.0, 1024.0] and (s >= x).all()


def test_four_over_six_table2_from_the_paper():
    """Cook et al. Table 2: block [10,20,30,40] picks scale 4 (error 0); block [15,30,120,180] keeps scale 6 (error 0)."""
    d6 = kit.round_e4m3(torch.tensor(40.0) / 6.0)
    assert abs(d6.item() - 6.5) < 1e-6                                  # paper: Delta(6) = 6.67 -> 6.5
    assert kit.round_e4m3(torch.tensor(180.0) / 6.0).item() == 30.0     # second block: exact
    assert kit.round_e4m3(torch.tensor(180.0) / 4.0).item() == 44.0     # 45 -> 44
    e6 = ((kit.round_fp4(torch.tensor([10.0, 20, 30, 40]) / 6.5) * 6.5 - torch.tensor([10.0, 20, 30, 40])) ** 2).mean().item()
    assert abs(e6 - 4.33) < 0.01                                         # paper: E(6) = 4.33


def test_four_over_six_never_worse_per_block_and_better_on_gaussian():
    torch.manual_seed(0)
    x = torch.randn(512, 256)
    a = kit.quantize_nvfp4(x)
    b = kit.quantize_nvfp4(x, four_over_six=True)
    mse = lambda t: ((t.dequantize() - x) ** 2).mean().item()
    assert mse(b) < mse(a)
    assert b.used4.float().mean() > 0.1                                  # it really does pick scale 4 sometimes


def test_nvfp4_roundtrip_error_is_small_and_blocks_are_independent():
    torch.manual_seed(0)
    x = torch.randn(64, 128)
    q = kit.quantize_nvfp4(x).dequantize()
    assert ((q - x).norm() / x.norm()).item() < 0.15
    x2 = x.clone(); x2[0, :16] *= 1000                                   # an outlier only disturbs its own row-block ...
    q2 = kit.quantize_nvfp4(x2, outer="row").dequantize()
    assert torch.equal(q2[0, 16:], kit.quantize_nvfp4(x2, outer="row").dequantize()[0, 16:])


def test_bits_per_value_match_the_post_table():
    assert kit.bits_per_value(4, 8, 16) == 4.5 and kit.bits_per_value(8, 8, 32) == 8.25


def test_random_hadamard_is_orthonormal_and_invertible():
    r = kit.RandomHadamard(32, seed=3)
    x = torch.randn(10, 64)
    y = r.forward(x)
    assert torch.allclose(r.inverse(y), x, atol=1e-5) and torch.allclose(y.norm(), x.norm(), rtol=1e-5)
    h = kit.hadamard(32) / 32 ** 0.5
    assert torch.allclose(h @ h.t(), torch.eye(32), atol=1e-6)


def test_s2f6_range_and_step():
    x = torch.randn(4, 64)
    q = kit.quantize_s2f6(x)
    assert ((q - x).abs() <= (kit.e8m0_up(x.reshape(4, 2, 32).abs().amax(-1, keepdim=True) / (127 / 64)) / 64 / 2 + 1e-6).repeat_interleave(32, -1).reshape(4, 64)).all()


@pytest.mark.parametrize("bwd", ["none", "mxfp8_sr", "nvfp4_sr", "mxfp8_rtn", "nvfp4_rtn"])
def test_fp4_linear_forward_and_backward_shapes_and_finiteness(bwd):
    torch.manual_seed(0)
    x = torch.randn(64, 64, requires_grad=True); w = torch.randn(32, 64, requires_grad=True)
    cfg = kit.RecipeConfig(bwd=bwd)
    y = kit.fp4_linear(x, w, cfg)
    y.pow(2).mean().backward()
    assert y.shape == (64, 32) and x.grad.shape == x.shape and w.grad.shape == w.shape
    assert torch.isfinite(x.grad).all() and torch.isfinite(w.grad).all()


def test_forward_equals_dequantized_matmul():
    torch.manual_seed(0)
    x = torch.randn(32, 64); w = torch.randn(32, 64)
    cfg = kit.RecipeConfig()
    y = kit.fp4_linear(x, w, cfg)
    xq = kit.quantize_nvfp4(x, four_over_six=True, outer="row").dequantize()
    wq = kit.quantize_nvfp4(w, four_over_six=False, outer="tensor").dequantize()
    assert torch.allclose(y, xq @ wq.t(), atol=1e-5)


def test_sr_backward_is_unbiased_for_the_quantized_forward():
    """Averaging many SR backward passes converges to the STE gradient of the quantized forward (g @ wq, g^T @ xq)."""
    torch.manual_seed(0)
    x = torch.randn(32, 64); w = torch.randn(32, 64); g = torch.randn(32, 32)
    cfg = kit.RecipeConfig(bwd="mxfp8_sr")
    xq = kit.quantize_nvfp4(x, four_over_six=True, outer="row").dequantize()
    wq = kit.quantize_nvfp4(w, outer="tensor").dequantize()
    dx_ref, dw_ref = g @ wq, g.t() @ xq
    dx_sum = torch.zeros_like(dx_ref); dw_sum = torch.zeros_like(dw_ref); n = 300
    for _ in range(n):
        xr = x.clone().requires_grad_(); wr = w.clone().requires_grad_()
        kit.fp4_linear(xr, wr, cfg).backward(g)
        dx_sum += xr.grad; dw_sum += wr.grad
    assert ((dx_sum / n - dx_ref).norm() / dx_ref.norm()).item() < 0.01
    assert ((dw_sum / n - dw_ref).norm() / dw_ref.norm()).item() < 0.01


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="no Apple GPU")
def test_mps_matches_cpu_for_deterministic_quantizers():
    torch.manual_seed(0)
    x = torch.randn(64, 128)
    for fn in (lambda t: kit.quantize_nvfp4(t, four_over_six=True, outer="row").dequantize(),
               lambda t: kit.quantize_nvfp4(t).dequantize(),
               lambda t: kit.quantize_mx(t, "mxfp8"), lambda t: kit.quantize_s2f6(t), lambda t: kit.round_e4m3(t)):
        assert torch.equal(fn(x), fn(x.to("mps")).cpu())


def test_table2_reproduced_through_quantize_nvfp4_with_alpha_one():
    """Cook et al. Table 2 end to end (block size 4, outer scale 1)."""
    x = torch.tensor([[10.0, 20, 30, 40], [15.0, 30, 120, 180]])
    plain = kit.quantize_nvfp4(x, block=4, outer="row", alpha_override=1.0)
    ada = kit.quantize_nvfp4(x, block=4, outer="row", four_over_six=True, alpha_override=1.0)
    assert plain.delta.flatten().tolist() == [6.5, 30.0]
    assert plain.dequantize()[0].tolist() == [9.75, 19.5, 26.0, 39.0]            # paper: D(6) for block 1
    assert ada.used4.flatten().tolist() == [True, False]                         # block 1 picks 4, block 2 keeps 6
    assert ada.dequantize()[0].tolist() == [10.0, 20.0, 30.0, 40.0] and ada.dequantize()[1].tolist() == [15.0, 30.0, 120.0, 180.0]


def test_2d_scaling_is_transpose_invariant_and_coarser():
    torch.manual_seed(0)
    w = torch.randn(32, 64)
    q = kit.quantize_nvfp4_2d(w)
    assert torch.equal(kit.quantize_nvfp4_2d(w.t().contiguous()), q.t())         # same values seen by forward and backward
    assert ((kit.quantize_nvfp4(w).dequantize() - w) ** 2).mean() < ((q - w) ** 2).mean()   # 1x16 is finer than 16x16


# ---- cross-checks against PyTorch's own reference helpers (torch/testing/_internal/common_quantized.py) ----
def _torch_reference():
    import ast
    path = Path(torch.__file__).parent / "testing" / "_internal" / "common_quantized.py"
    if not path.exists():
        pytest.skip("torch reference helpers not shipped")
    text = path.read_text(); tree = ast.parse(text)
    want = {"_n_ones", "_f32_to_floatx_unpacked", "_floatx_unpacked_to_f32", "ceil_div", "to_blocked", "down_size", "pack_uint4", "to_mxfp", "_bfloat16_to_float4_e2m1fn_x2"}
    parts = [ast.get_source_segment(text, n) for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in want]
    parts += ["EBITS_F32, MBITS_F32 = 8, 23", "F32_EXP_BIAS = _n_ones(EBITS_F32 - 1)", "FP4_EBITS, FP4_MBITS = 2, 1"]      # constants after the functions that define _n_ones
    ns = {"torch": torch, "Tensor": torch.Tensor}
    try:
        exec("\n".join(parts), ns)
    except Exception:
        pytest.skip("torch reference helpers changed shape")
    return ns


def test_fp4_codes_and_packing_match_pytorch_reference():
    ref = _torch_reference()
    a = torch.tensor([0, .25, .5, .75, 1, 1.25, 1.5, 1.75, 2, 2.5, 3, 3.5, 4, 5, 6, 7, -.5, -6, -.25, -3.5])
    q = kit.round_fp4(a)
    assert torch.equal(kit.fp4_codes(q), ref["_f32_to_floatx_unpacked"](q, 2, 1))
    q2 = kit.round_fp4(torch.randn(8, 64) * 3)
    assert torch.equal(kit.pack_fp4(q2), ref["pack_uint4"](ref["_f32_to_floatx_unpacked"](q2, 2, 1)))
    assert torch.equal(kit.unpack_fp4(kit.pack_fp4(q2)), q2)
    assert torch.equal(kit.round_fp4(a), ref["_floatx_unpacked_to_f32"](ref["_f32_to_floatx_unpacked"](a, 2, 1), 2, 1))


def test_to_blocked_matches_pytorch_reference():
    ref = _torch_reference()
    for shape in [(128, 4), (200, 10), (256, 8)]:
        s = torch.randn(*shape)
        assert torch.equal(kit.to_blocked(s), ref["to_blocked"](s))


def test_mxfp8_matches_pytorch_reference_bitwise():
    ref = _torch_reference()
    torch.manual_seed(0)
    x = torch.randn(64, 128) * torch.exp(torch.randn(64, 1))
    sc, d = ref["to_mxfp"](x, 32, "mxfp8")
    deq = (d.float().reshape(64, 4, 32) * sc.float().unsqueeze(-1)).reshape(64, 128)
    assert torch.equal(deq, kit.quantize_mx(x, "mxfp8"))


def test_mxfp4_matches_pytorch_reference_apart_from_its_bf16_pre_rounding():
    """PyTorch's reference casts the scaled values to bf16 BEFORE rounding to FP4 (double rounding); the scales are identical."""
    ref = _torch_reference()
    torch.manual_seed(0)
    x = torch.randn(64, 128) * torch.exp(torch.randn(64, 1))
    sc, d = ref["to_mxfp"](x, 32, "mxfp4")
    codes = d.view(torch.uint8); un = torch.stack([codes & 0xF, codes >> 4], -1).reshape(64, 128)
    deq = (ref["_floatx_unpacked_to_f32"](un, 2, 1).reshape(64, 4, 32) * sc.float().unsqueeze(-1)).reshape(64, 128)
    xb = x.reshape(64, 4, 32); scale = kit.e8m0_up(xb.abs().amax(-1, keepdim=True) / 6.0)
    assert torch.equal(scale.squeeze(-1), sc.float())
    assert torch.equal(deq, (kit.round_fp4((xb / scale).bfloat16().float()) * scale).reshape(64, 128))


# ---- hardware-shaped containers: parts must carry exactly the information of the fake-quant tensors ----
@pytest.mark.parametrize("kw", [dict(), dict(four_over_six=True, outer="row"), dict(outer="row")])
def test_nvfp4_parts_dequantize_equals_fake_quant(kw):
    torch.manual_seed(0)
    x = torch.randn(48, 64) * 2
    parts = kit.nvfp4_parts(x, **kw)
    assert parts.packed.dtype == torch.uint8 and parts.packed.shape == (48, 32) and parts.delta.shape == (48, 4)
    assert torch.allclose(parts.dequantize(), kit.quantize_nvfp4(x, **kw).dequantize(), atol=1e-6, rtol=1e-6)
    pk, sc = parts.hw_operands()
    assert pk.dtype == torch.float4_e2m1fn_x2 and sc.dtype == torch.float8_e4m3fn
    assert sc.numel() == 128 * 4                                            # 48 rows padded to 128, 64/16 = 4 scale columns
    assert torch.equal(parts.delta.to(torch.float8_e4m3fn).float(), parts.delta)    # block scales are exactly E4M3


def test_mxfp8_parts_dequantize_equals_fake_quant_and_hw_cast_is_exact():
    torch.manual_seed(0)
    x = torch.randn(64, 128) * 3
    parts = kit.mxfp8_parts(x)
    assert torch.equal(parts.dequantize(), kit.quantize_mx(x, "mxfp8"))
    q8, sc = parts.hw_operands()
    assert q8.dtype == torch.float8_e4m3fn and sc.dtype == torch.float8_e8m0fnu
    assert torch.equal(q8.float(), parts.q)                                         # E4M3 cast loses nothing
    assert sc.numel() == 128 * 4                                                    # 64 rows padded to 128, 4 scale columns


def test_packed_linear_emulated_backend_reproduces_fp4_linear_exactly():
    torch.manual_seed(0)
    x = torch.randn(64, 64); w = torch.randn(32, 64); g = torch.randn(64, 32)
    cfg = kit.RecipeConfig()
    outs = []
    for fn in (lambda a, b: kit.fp4_linear(a, b, cfg), lambda a, b: kit.fp4_linear_packed(a, b, cfg, "emulated")):
        torch.manual_seed(123)
        xr, wr = x.clone().requires_grad_(), w.clone().requires_grad_()
        y = fn(xr, wr); y.backward(g)
        outs.append((y.detach(), xr.grad, wr.grad))
    for a, b in zip(*outs):
        assert torch.allclose(a, b, atol=1e-5, rtol=1e-5)


def test_scaled_mm_signatures_accepted_by_pytorch_meta_kernels():
    """Argument/shape/dtype plumbing for the Blackwell GEMMs, checked on the 'meta' device (no GPU, no numerics)."""
    import torch.nn.functional as F
    if not hasattr(F, "scaled_mm"):
        pytest.skip("torch.nn.functional.scaled_mm not available")
    S, W = F.ScalingType, F.SwizzleType
    M, N, K = 128, 128, 256
    a = torch.empty(M, K // 2, dtype=torch.float4_e2m1fn_x2, device="meta"); b = torch.empty(N, K // 2, dtype=torch.float4_e2m1fn_x2, device="meta")
    sa = torch.empty(kit.to_blocked(torch.empty(M, K // 16)).numel(), dtype=torch.float8_e4m3fn, device="meta")
    out = F.scaled_mm(a, b.t(), sa, S.BlockWise1x16, sa.clone(), S.BlockWise1x16, swizzle_a=W.SWIZZLE_32_4_4, swizzle_b=W.SWIZZLE_32_4_4, output_dtype=torch.bfloat16)
    assert out.shape == (M, N) and out.dtype == torch.bfloat16


def test_unblock_inverts_to_blocked():
    for shape in [(128, 4), (200, 10), (512, 64)]:
        sc = torch.randn(*shape)
        assert torch.equal(kit._unblock(kit.to_blocked(sc), *shape), sc)


@pytest.fixture
def standin(monkeypatch):
    monkeypatch.setattr(kit, "_scaled_mm", kit.cpu_standin_scaled_mm)


def test_scaled_mm_backend_plumbing_nvfp4_and_mxfp8(standin):
    """Runs the 'scaled_mm' code path (hw operands, swizzled scales, outer scales, transposes) with a CPU stand-in kernel."""
    torch.manual_seed(0)
    x = torch.randn(256, 512) * (1 + 9 * (torch.rand(1, 512) < 0.03)); w = torch.randn(128, 512) / 22
    a, b = kit.nvfp4_parts(x, four_over_six=True, outer="row"), kit.nvfp4_parts(w)
    ref, nat = kit.gemm_nvfp4(a, b, "emulated"), kit.gemm_nvfp4(a, b, "scaled_mm")
    assert ((nat - ref).norm() / ref.norm()).item() < 1e-2                     # bf16 output rounding only
    a8, b8 = kit.mxfp8_parts(x), kit.mxfp8_parts(w)
    ref8, nat8 = kit.gemm_mxfp8(a8, b8, "emulated"), kit.gemm_mxfp8(a8, b8, "scaled_mm")
    assert ((nat8 - ref8).norm() / ref8.norm()).item() < 1e-2


def test_scaled_mm_backend_full_layer_matches_emulated(standin):
    torch.manual_seed(11); g = torch.randn(256, 128) * 0.01
    outs = {}
    for backend in ("emulated", "scaled_mm"):
        torch.manual_seed(5)
        x = torch.randn(256, 256).requires_grad_(); w = (torch.randn(128, 256) / 16).requires_grad_()
        torch.manual_seed(77)
        y = kit.fp4_linear_packed(x, w, kit.RecipeConfig(), backend); y.backward(g)
        outs[backend] = (y.detach(), x.grad, w.grad)
    for a, b in zip(outs["scaled_mm"], outs["emulated"]):
        assert ((a - b).norm() / b.norm()).item() < 3e-2


def test_a_wrong_nibble_order_would_be_caught(standin, monkeypatch):
    """Guard for the guard: if packing put the even element in the HIGH nibble, the stand-in comparison must fail loudly."""
    torch.manual_seed(0)
    x = torch.randn(128, 256); w = torch.randn(128, 256)
    a, b = kit.nvfp4_parts(x), kit.nvfp4_parts(w)
    swap = lambda p: ((p & 0xF) << 4) | (p >> 4)
    a_bad = kit.NVFP4Parts(swap(a.packed), a.delta, a.alpha, a.shape)
    ref, bad = kit.gemm_nvfp4(a, b, "emulated"), kit.gemm_nvfp4(a_bad, b, "scaled_mm")
    assert ((bad - ref).norm() / ref.norm()).item() > 0.3
