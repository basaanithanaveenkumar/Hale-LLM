"""fp4kit - emulated (fake-quant) reference numerics for 4-bit training.

Everything here runs on plain PyTorch tensors (cpu / mps / cuda): values are *rounded to the grid of the target format*
and kept in float32, so the arithmetic is exact emulation of the number format but NOT a speed-up. Real FP4 tensor-core
kernels need an NVIDIA Blackwell GPU (see the Blackwell notebook).

Sources the definitions follow
  * NVFP4 / two-level scaling, 2D weight scaling, RHT on Wgrad, stochastic rounding of gradients:
    NVIDIA, "Pretraining Large Language Models with NVFP4", arXiv:2509.25149 (Appendix B).
  * Four Over Six (4/6) adaptive block scaling, Table 2 and Sec. 3: Cook et al., arXiv:2512.02010.
  * The recipe (NVFP4 forward, MXFP8 + stochastic-rounding backward, RHT + S2F6 combine):
    Thinking Machines Lab, "End-to-End 4-Bit Training" (blog post).
Where the sources leave a detail open, the choice made here is marked ASSUMPTION.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn

__all__ = [
    "FP4_GRID", "round_fp4", "round_e4m3", "e8m0_up", "NVFP4Tensor", "quantize_nvfp4", "quantize_nvfp4_2d", "quantize_mx", "quantize_int4_block",
    "hadamard", "RandomHadamard", "quantize_s2f6", "NVFP4Parts", "MXFP8Parts", "nvfp4_parts", "mxfp8_parts", "gemm_nvfp4", "gemm_mxfp8", "fp4_linear_packed", "cpu_standin_scaled_mm", "fp4_codes", "pack_fp4", "unpack_fp4", "to_blocked", "RecipeConfig", "fp4_linear", "Fp4Linear", "bits_per_value",
]

FP4_GRID = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])        # the 8 magnitudes of E2M1
E2M1_MAX, E4M3_MAX = 6.0, 448.0
_TINY = 1e-30


def _uniform(shape, like: torch.Tensor, gen: torch.Generator | None) -> torch.Tensor:
    if gen is None:
        return torch.rand(shape, device=like.device, dtype=like.dtype)
    return torch.rand(shape, generator=gen, device=gen.device, dtype=like.dtype).to(like.device)


# ---------------------------------------------------------------- element rounding
def round_fp4(a: torch.Tensor, rounding: str = "rtn", gen: torch.Generator | None = None) -> torch.Tensor:
    """Round already-scaled values to the E2M1 grid {0, .5, 1, 1.5, 2, 3, 4, 6} (and negatives); saturates at +-6.

    'rtn' = round to nearest, ties to the even code (what the hardware cvt.rn does).
    'sr'  = stochastic rounding: up with probability (a - lo) / (hi - lo), so E[round(a)] = a inside [-6, 6].
    Written with comparisons and index lookups only, so it runs identically on cpu / mps / cuda.
    """
    g = FP4_GRID.to(device=a.device, dtype=a.dtype)
    m = a.abs().clamp(max=E2M1_MAX)
    if rounding == "rtn":
        # midpoints .25 .75 1.25 1.75 2.5 3.5 5 ; a tie goes to the neighbour whose 3-bit code is even
        idx = ((m > 0.25).long() + (m >= 0.75).long() + (m > 1.25).long() + (m >= 1.75).long()
               + (m > 2.5).long() + (m >= 3.5).long() + (m > 5.0).long())
    elif rounding == "sr":
        lo_idx = sum((m >= g[k]).long() for k in range(1, 8))
        hi_idx = (lo_idx + 1).clamp(max=7)
        lo, hi = g[lo_idx], g[hi_idx]
        p_up = torch.where(hi > lo, (m - lo) / (hi - lo).clamp(min=_TINY), torch.zeros_like(m))
        idx = lo_idx + (_uniform(m.shape, m, gen) < p_up).long()
    else:
        raise ValueError(rounding)
    return torch.sign(a) * g[idx]


def round_e4m3(x: torch.Tensor, rounding: str = "rtn", gen: torch.Generator | None = None) -> torch.Tensor:
    """Round to the FP8 E4M3 grid (3 mantissa bits, min normal 2^-6, subnormal step 2^-9, max 448), saturating.

    Pure arithmetic (no float8 dtype), so it also works on mps, which has no float8 support.
    """
    a = x.abs().clamp(max=E4M3_MAX)
    e = torch.floor(torch.log2(a.clamp(min=2.0 ** -9))).clamp(min=-6.0)
    step = torch.exp2(e - 3.0)
    q = a / step
    q = torch.round(q) if rounding == "rtn" else torch.floor(q + _uniform(q.shape, q, gen))
    return torch.sign(x) * (q * step).clamp(max=E4M3_MAX)


def e8m0_up(x: torch.Tensor) -> torch.Tensor:
    """Round a positive scale UP to the next power of two (E8M0 stores only an exponent). Non-positive -> 1."""
    e = torch.ceil(torch.log2(x.clamp(min=_TINY))).clamp(-127.0, 127.0)
    return torch.where(x > 0, torch.exp2(e), torch.ones_like(x))


# ---------------------------------------------------------------- NVFP4
@dataclass
class NVFP4Tensor:
    """x ~= q * delta * alpha, with q on the FP4 grid, delta an E4M3 block scale (per 16 values), alpha an FP32 outer scale."""
    q: torch.Tensor        # (..., nblocks, block) values on the FP4 grid
    delta: torch.Tensor    # (..., nblocks, 1) E4M3-rounded block scales
    alpha: torch.Tensor    # outer scale, broadcastable to q
    shape: tuple
    used4: torch.Tensor | None = None   # which blocks chose the "map max to 4" scale (4/6 only)

    def dequantize(self) -> torch.Tensor:
        return (self.q * self.delta * self.alpha).reshape(self.shape)


def quantize_nvfp4(x: torch.Tensor, *, block: int = 16, four_over_six: bool = False, rounding: str = "rtn",
                   outer: str = "tensor", gen: torch.Generator | None = None,
                   alpha_override: float | None = None) -> NVFP4Tensor:
    """NVFP4 quantization along the LAST axis (transpose first to block along another axis).

    outer = 'tensor': one FP32 scale for the whole tensor (NVIDIA recipe);  'row': one per row, i.e. per token (the blog's activations).
    alpha = amax / (6 * M_fp8) with M_fp8 = 448 (256 when 4/6 is on, so a block can still pick scale 4 without overflowing E4M3).
    four_over_six: quantize each block with its max mapped to 6 AND to 4, keep the lower-MSE version (strict '<' for 4).
    alpha_override: override the outer scale with a fixed number (used to reproduce the 4/6 paper's hand-worked Table 2, where alpha = 1).
    """
    if four_over_six and rounding != "rtn":
        raise ValueError("4/6 selects by round-to-nearest error; combine it with 'rtn' only")
    x = x.float(); shape = tuple(x.shape); k = shape[-1]
    if k % block: raise ValueError(f"last dim {k} not a multiple of {block}")
    amax_t = x.abs().max().reshape([1] * x.ndim) if outer == "tensor" else x.abs().amax(-1, keepdim=True)
    m_fp8 = 256.0 if four_over_six else E4M3_MAX
    alpha = (amax_t / (E2M1_MAX * m_fp8)).clamp(min=_TINY).unsqueeze(-1)
    if alpha_override is not None:
        alpha = torch.full_like(alpha, float(alpha_override))
    xb = x.reshape(*shape[:-1], k // block, block)
    amax_b = xb.abs().amax(-1, keepdim=True)

    def attempt(m_target: float):
        delta = round_e4m3(amax_b / (alpha * m_target))
        safe = torch.where(delta > 0, delta, torch.ones_like(delta))
        q = round_fp4(xb / (alpha * safe), rounding, gen) * (delta > 0)
        err = ((q * delta * alpha - xb) ** 2).mean(-1, keepdim=True)
        return delta, q, err

    d6, q6, e6 = attempt(6.0)
    if not four_over_six:
        return NVFP4Tensor(q6, d6, alpha, shape)
    d4, q4, e4 = attempt(4.0)
    use4 = e4 < e6
    return NVFP4Tensor(torch.where(use4, q4, q6), torch.where(use4, d4, d6), alpha, shape, use4.squeeze(-1))


def quantize_nvfp4_2d(x: torch.Tensor, *, tile: int = 16) -> torch.Tensor:
    """NVFP4 with one E4M3 scale per tile x tile square (NVIDIA's 2D weight scaling): the quantized weight is the same
    whether you block along rows or columns, i.e. identical in the forward and the transposed backward use. Returns dequantized."""
    x = x.float(); r, c = x.shape
    if r % tile or c % tile: raise ValueError("both dims must be multiples of tile")
    t = x.reshape(r // tile, tile, c // tile, tile)
    amax_t = x.abs().max()
    alpha = (amax_t / (E2M1_MAX * E4M3_MAX)).clamp(min=_TINY)
    amax_b = t.abs().amax((1, 3), keepdim=True)
    delta = round_e4m3(amax_b / (alpha * E2M1_MAX))
    safe = torch.where(delta > 0, delta, torch.ones_like(delta))
    q = round_fp4(t / (alpha * safe)) * (delta > 0)
    return (q * delta * alpha).reshape(r, c)


# ---------------------------------------------------------------- MX formats, int4 baseline
def _mx_quant(x: torch.Tensor, fmt: str, block: int, rounding: str, gen, scale_rule: str):
    """-> (q, scale): q on the element grid with shape (..., K/block, block); scale (..., K/block, 1) a power of two."""
    x = x.float(); shape = x.shape; k = shape[-1]
    if k % block: raise ValueError(f"last dim {k} not a multiple of {block}")
    xb = x.reshape(*shape[:-1], k // block, block)
    amax = xb.abs().amax(-1, keepdim=True)
    emax, elem_max = (8, E4M3_MAX) if fmt == "mxfp8" else (2, E2M1_MAX)
    if scale_rule == "ceil":
        scale = e8m0_up(amax / elem_max)
    else:
        scale = torch.where(amax > 0, torch.exp2(torch.floor(torch.log2(amax.clamp(min=_TINY))) - emax), torch.ones_like(amax))
    s = xb / scale
    q = round_e4m3(s, rounding, gen) if fmt == "mxfp8" else round_fp4(s, rounding, gen)
    return q, scale


def quantize_mx(x: torch.Tensor, fmt: str = "mxfp8", *, block: int = 32, rounding: str = "rtn",
                gen: torch.Generator | None = None, scale_rule: str = "ceil") -> torch.Tensor:
    """OCP microscaling fake-quant along the last axis; returns the DEQUANTIZED tensor. fmt: 'mxfp8' (E4M3 elements) or 'mxfp4'.

    Scale = power of two (E8M0). 'ceil' rounds it up so the block max never saturates (recommended by Mishra et al.);
    'floor' is the literal OCP rule floor(log2 amax) - emax. ASSUMPTION: MXFP8 uses E4M3 elements everywhere.
    """
    q, scale = _mx_quant(x, fmt, block, rounding, gen, scale_rule)
    return (q * scale).reshape(x.shape)


def quantize_int4_block(x: torch.Tensor, *, block: int = 16) -> torch.Tensor:
    """Symmetric INT4 with a float absmax/7 scale per block (comparison baseline)."""
    x = x.float(); shape = x.shape
    xb = x.reshape(*shape[:-1], shape[-1] // block, block)
    s = (xb.abs().amax(-1, keepdim=True) / 7.0).clamp(min=_TINY)
    return (torch.round(xb / s).clamp(-7, 7) * s).reshape(shape)



# ---------------------------------------------------------------- packing and scale layout for real FP4 GEMMs
def fp4_codes(q: torch.Tensor) -> torch.Tensor:
    """Values on the FP4 grid -> 4-bit E2M1 codes (bit 3 = sign, bits 0-2 = index into FP4_GRID), as uint8."""
    g = FP4_GRID.to(q.device, q.dtype)
    idx = (q.abs().unsqueeze(-1) == g).float().argmax(-1)
    neg = (torch.signbit(q)).to(torch.uint8)
    return (neg << 3) | idx.to(torch.uint8)


def pack_fp4(q: torch.Tensor) -> torch.Tensor:
    """Pack FP4-grid values two per byte along the last axis: byte = (code[2i+1] << 4) | code[2i]  (even element in the LOW nibble,
    the layout PyTorch's float4_e2m1fn_x2 reference helpers use). Result is uint8 with last dim halved."""
    c = fp4_codes(q)
    if c.shape[-1] % 2: raise ValueError("last dim must be even")
    return (c[..., 1::2] << 4) | c[..., 0::2]


def unpack_fp4(packed: torch.Tensor) -> torch.Tensor:
    """Inverse of pack_fp4 -> float32 values on the FP4 grid."""
    lo, hi = packed & 0xF, packed >> 4
    codes = torch.stack([lo, hi], -1).reshape(*packed.shape[:-1], packed.shape[-1] * 2)
    g = FP4_GRID.to(packed.device)
    return torch.where((codes & 8).bool(), -1.0, 1.0) * g[(codes & 7).long()]


def to_blocked(scales: torch.Tensor) -> torch.Tensor:
    """Rearrange a 2-D scale matrix (rows, K/block) into the 'swizzled' layout cuBLAS block-scaled GEMMs expect
    (128 x 4 tiles stored as 32 x 16; zero-padded to multiples of 128 rows and 4 columns). Returns a flat tensor.
    Same permutation as PyTorch's own test helper `to_blocked`."""
    rows, cols = scales.shape
    nr, nc = -(-rows // 128), -(-cols // 4)
    padded = scales
    if (rows, cols) != (nr * 128, nc * 4):
        padded = torch.zeros((nr * 128, nc * 4), dtype=scales.dtype, device=scales.device)
        padded[:rows, :cols] = scales
    blocks = padded.view(nr, 128, nc, 4).permute(0, 2, 1, 3)
    return blocks.reshape(-1, 4, 32, 4).transpose(1, 2).reshape(-1, 32, 16).flatten()


# ---------------------------------------------------------------- RHT + S2F6 (communication)
def hadamard(n: int, device=None) -> torch.Tensor:
    if n & (n - 1): raise ValueError("n must be a power of two")
    h = torch.ones(1, 1, device=device)
    while h.shape[0] < n:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    return h


class RandomHadamard:
    """Orthonormal random Hadamard transform on blocks of `block` values: y = (x * s) H / sqrt(block), s = random +-1 signs."""

    def __init__(self, block: int = 32, seed: int = 0):
        g = torch.Generator().manual_seed(seed)
        self.block = block
        self.signs = (torch.randint(0, 2, (block,), generator=g).float() * 2 - 1)
        self.h = hadamard(block) / math.sqrt(block)

    def _to(self, x):
        return self.signs.to(x.device), self.h.to(x.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s, h = self._to(x); shape = x.shape
        return ((x.reshape(*shape[:-1], -1, self.block) * s) @ h).reshape(shape)

    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        s, h = self._to(y); shape = y.shape
        return ((y.reshape(*shape[:-1], -1, self.block) @ h) * s).reshape(shape)


def quantize_s2f6(x: torch.Tensor, *, block: int = 32) -> torch.Tensor:
    """S2F6 = signed fixed point, 2 integer + 6 fractional bits (range -128/64 .. 127/64), one E8M0 scale per `block`.

    ASSUMPTION: the post does not say how the scale is chosen; here scale = smallest power of two with amax/scale <= 127/64.
    Returns the dequantized tensor.
    """
    x = x.float(); shape = x.shape
    xb = x.reshape(*shape[:-1], shape[-1] // block, block)
    scale = e8m0_up(xb.abs().amax(-1, keepdim=True) / (127.0 / 64.0))
    q = torch.round(xb / scale * 64.0).clamp(-128, 127)
    return (q / 64.0 * scale).reshape(shape)


def bits_per_value(element_bits: int, scale_bits: int, block: int) -> float:
    return element_bits + scale_bits / block


# ---------------------------------------------------------------- hardware-shaped containers and GEMMs
@dataclass
class NVFP4Parts:
    """What an NVFP4 GEMM consumes for one 2-D operand (rows, K): packed FP4, E4M3 block scales, FP32 outer scale."""
    packed: torch.Tensor    # uint8 (rows, K/2), two FP4 codes per byte (even element in the low nibble)
    delta: torch.Tensor     # float32 (rows, K/16), values exactly representable in E4M3
    alpha: torch.Tensor     # float32, one number (outer='tensor') or (rows,) (outer='row')
    shape: tuple

    def dequantize(self) -> torch.Tensor:
        q = unpack_fp4(self.packed).reshape(self.shape[0], -1, 16)
        a = self.alpha.reshape(-1, 1, 1) if self.alpha.numel() > 1 else self.alpha.reshape(1, 1, 1)
        return (q * self.delta.unsqueeze(-1) * a).reshape(self.shape)

    def hw_operands(self):
        """(packed as torch.float4_e2m1fn_x2, swizzled E4M3 block scales) - the tensors handed to a Blackwell GEMM."""
        return self.packed.contiguous().view(torch.float4_e2m1fn_x2), to_blocked(self.delta.to(torch.float8_e4m3fn))


@dataclass
class MXFP8Parts:
    q: torch.Tensor         # float32 values on the E4M3 grid (rows, K)
    scale: torch.Tensor     # float32 powers of two (rows, K/32)

    @property
    def shape(self):
        return tuple(self.q.shape)

    def dequantize(self) -> torch.Tensor:
        r, kk = self.q.shape
        return (self.q.reshape(r, -1, 32) * self.scale.unsqueeze(-1)).reshape(r, kk)

    def hw_operands(self):
        """(E4M3 elements, swizzled E8M0 block scales)."""
        biased = (torch.log2(self.scale).round() + 127).clamp(0, 254).to(torch.uint8)
        return self.q.to(torch.float8_e4m3fn).contiguous(), to_blocked(biased.view(torch.float8_e8m0fnu))


def nvfp4_parts(x: torch.Tensor, **kw) -> NVFP4Parts:
    """2-D x (rows, K) -> NVFP4Parts. kw as quantize_nvfp4 (four_over_six, rounding, outer, gen)."""
    qt = quantize_nvfp4(x, **kw)
    return NVFP4Parts(pack_fp4(qt.q.reshape(x.shape)), qt.delta.squeeze(-1), qt.alpha.reshape(-1), tuple(x.shape))


def mxfp8_parts(x: torch.Tensor, *, rounding: str = "rtn", gen=None) -> MXFP8Parts:
    q, scale = _mx_quant(x, "mxfp8", 32, rounding, gen, "ceil")
    return MXFP8Parts(q.reshape(x.shape), scale.squeeze(-1))


def _scaled_mm(a, sa, ra: str, b, sb, rb: str):
    """Block-scaled GEMM a @ b.T on a Blackwell GPU through PyTorch. ra/rb: '1x16' (NVFP4) or '1x32' (MXFP8).
    Tries torch.nn.functional.scaled_mm (recipes + swizzle enums), then the older torch._scaled_mm. NOT runnable off-GPU."""
    import torch.nn.functional as F
    if hasattr(F, "scaled_mm"):
        S, W = F.ScalingType, F.SwizzleType
        rec = {"1x16": S.BlockWise1x16, "1x32": S.BlockWise1x32}
        return F.scaled_mm(a, b.t(), sa, rec[ra], sb, rec[rb], swizzle_a=W.SWIZZLE_32_4_4, swizzle_b=W.SWIZZLE_32_4_4, output_dtype=torch.bfloat16)
    return torch._scaled_mm(a, b.t(), sa, sb, out_dtype=torch.bfloat16)


def _unblock(flat: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
    """Inverse of to_blocked: flat swizzled scales -> (rows, cols)."""
    nr, nc = -(-rows // 128), -(-cols // 4)
    t = flat.reshape(nr * nc, 32, 4, 4).transpose(1, 2)               # (tile, a, b, c)
    t = t.reshape(nr, nc, 128, 4).permute(0, 2, 1, 3).reshape(nr * 128, nc * 4)
    return t[:rows, :cols]


def cpu_standin_scaled_mm(a, sa, ra: str, b, sb, rb: str):
    """A CPU stand-in for `_scaled_mm`: decodes the SAME hardware-shaped operands (packed fp4 bytes / fp8 elements, swizzled scales)
    and multiplies in float32, returning bf16. It exercises all of our operand formatting, transposes and scale handling; it does NOT
    exercise the GPU kernel, so agreement with it proves consistency of our code, not correctness of cuBLAS."""
    def decode(t, sc, r):
        if r == "1x16":
            vals, blk = unpack_fp4(t.view(torch.uint8)), 16
        else:
            vals, blk = t.float(), 32
        rows, kk = vals.shape
        scale = _unblock(sc.float(), rows, kk // blk)
        return (vals.reshape(rows, -1, blk) * scale.unsqueeze(-1)).reshape(rows, kk)
    return (decode(a, sa, ra) @ decode(b, sb, rb).t()).bfloat16()


def gemm_nvfp4(a: NVFP4Parts, b: NVFP4Parts, backend: str = "emulated") -> torch.Tensor:
    """y = a @ b.T in float32. backend 'emulated' = dequantize then matmul (any device); 'scaled_mm' = Blackwell tensor cores
    on the packed operands, with the FP32 outer scales applied to the output (exact: they factor out of the sum)."""
    if backend == "emulated":
        return a.dequantize() @ b.dequantize().t()
    pa, sa = a.hw_operands(); pb, sb = b.hw_operands()
    y = _scaled_mm(pa, sa, "1x16", pb, sb, "1x16").float()
    oa = a.alpha.reshape(-1, 1) if a.alpha.numel() > 1 else a.alpha.reshape(1, 1)
    ob = b.alpha.reshape(1, -1) if b.alpha.numel() > 1 else b.alpha.reshape(1, 1)
    return y * oa * ob


def gemm_mxfp8(a: MXFP8Parts, b: MXFP8Parts, backend: str = "emulated") -> torch.Tensor:
    if backend == "emulated":
        return a.dequantize() @ b.dequantize().t()
    qa, sa = a.hw_operands(); qb, sb = b.hw_operands()
    return _scaled_mm(qa, sa, "1x32", qb, sb, "1x32").float()


class _Fp4PartsLinearFn(torch.autograd.Function):
    """Same recipe as _Fp4LinearFn, but every GEMM goes through gemm_nvfp4 / gemm_mxfp8 on packed operands, so the matmuls can run
    natively (backend='scaled_mm'). With backend='emulated' it reproduces fp4_linear exactly (same random draws in the same order)."""

    @staticmethod
    def forward(ctx, x, w, cfg, backend):
        if cfg.fwd != "nvfp4" or cfg.bwd != "mxfp8_sr":
            raise NotImplementedError("the packed path implements the post's recipe only: fwd='nvfp4', bwd='mxfp8_sr'")
        xp = nvfp4_parts(x, four_over_six=cfg.act_four_over_six, outer=cfg.act_outer)
        wp = nvfp4_parts(w, four_over_six=cfg.weight_four_over_six, outer=cfg.weight_outer)
        ctx.xq, ctx.wq, ctx.backend = xp.dequantize(), wp.dequantize(), backend      # saved forward tensors, de-quantized for the transposes
        return gemm_nvfp4(xp, wp, backend)

    @staticmethod
    def backward(ctx, g):
        backend = ctx.backend
        dx = gemm_mxfp8(mxfp8_parts(g, rounding="sr"), mxfp8_parts(ctx.wq.t().contiguous(), rounding="sr"), backend)
        dw = gemm_mxfp8(mxfp8_parts(g.t().contiguous(), rounding="sr"), mxfp8_parts(ctx.xq.t().contiguous(), rounding="sr"), backend)
        return dx, dw, None, None


def fp4_linear_packed(x, w, cfg: "RecipeConfig | None" = None, backend: str = "emulated"):
    """The post's recipe with GEMMs on hardware-shaped operands. x (T,K), w (N,K); dims multiples of 32 (128 is safest on GPU)."""
    return _Fp4PartsLinearFn.apply(x, w, cfg or RecipeConfig(), backend)


# ---------------------------------------------------------------- the recipe as an autograd function
@dataclass(frozen=True)
class RecipeConfig:
    fwd: str = "nvfp4"                 # 'nvfp4' | 'mxfp8' | 'none'
    bwd: str = "mxfp8_sr"              # '<mxfp8|nvfp4>_<sr|rtn>' | 'none' (straight-through, high precision)
    act_four_over_six: bool = True     # the post: 4/6 on activations ...
    weight_four_over_six: bool = False  # ... not on weights
    act_outer: str = "row"             # per-token outer scale for activations
    weight_outer: str = "tensor"       # weights: one outer scale (per expert = per weight matrix)


def _q_fwd(t: torch.Tensor, cfg: RecipeConfig, *, is_weight: bool) -> torch.Tensor:
    if cfg.fwd == "none":
        return t
    if cfg.fwd == "mxfp8":
        return quantize_mx(t, "mxfp8", block=32)
    return quantize_nvfp4(t, four_over_six=cfg.weight_four_over_six if is_weight else cfg.act_four_over_six,
                          outer=cfg.weight_outer if is_weight else cfg.act_outer).dequantize()


def _q_bwd(t: torch.Tensor, cfg: RecipeConfig) -> torch.Tensor:
    """Re-quantize a tensor (blocks along its last axis) with stochastic rounding for the backward GEMMs."""
    fmt, rounding = cfg.bwd.rsplit("_", 1)                       # e.g. 'mxfp8_sr', 'nvfp4_rtn'
    if fmt == "mxfp8":
        return quantize_mx(t, "mxfp8", block=32, rounding=rounding)
    return quantize_nvfp4(t, rounding=rounding, outer="tensor").dequantize()


class _Fp4LinearFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w, cfg):
        xq, wq = _q_fwd(x, cfg, is_weight=False), _q_fwd(w, cfg, is_weight=True)    # blocks run along K (the contraction axis)
        ctx.save_for_backward(xq, wq); ctx.cfg = cfg
        return xq @ wq.t()

    @staticmethod
    def backward(ctx, g):
        xq, wq = ctx.saved_tensors; cfg = ctx.cfg
        if cfg.bwd == "none":
            return g @ wq, g.t() @ xq, None
        # DGRAD  dx = g @ wq        contracts over N -> blocks must run along N:  g (T,N) already is,  wq^T (K,N) is re-blocked
        dx = _q_bwd(g, cfg) @ _q_bwd(wq.t().contiguous(), cfg).t()
        # WGRAD  dw = g^T @ xq      contracts over T -> blocks along T:  g^T (N,T) and xq^T (K,T)
        dw = _q_bwd(g.t().contiguous(), cfg) @ _q_bwd(xq.t().contiguous(), cfg).t()
        return dx, dw, None


def fp4_linear(x: torch.Tensor, w: torch.Tensor, cfg: RecipeConfig) -> torch.Tensor:
    """y = x @ w^T with the recipe's quantizers. x: (T, K), w: (N, K); K and N and T must be multiples of 32."""
    return _Fp4LinearFn.apply(x, w, cfg)


class Fp4Linear(nn.Module):
    """nn.Linear without bias; fp32 master weight, quantized on the fly. Leading dims are flattened (T = prod(leading))."""

    def __init__(self, in_features: int, out_features: int, cfg: RecipeConfig | None = None):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features) / math.sqrt(in_features))
        self.cfg = cfg or RecipeConfig(fwd="none", bwd="none")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lead = x.shape[:-1]
        y = fp4_linear(x.reshape(-1, x.shape[-1]), self.weight, self.cfg)
        return y.reshape(*lead, -1)
