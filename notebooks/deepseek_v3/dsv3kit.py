"""dsv3kit - from-scratch building blocks for the DeepSeek-V3 notebooks (pure PyTorch, CPU/MPS/CUDA).

Every block is a small class that follows the paper's equations (arXiv 2412.19437).  The notebooks
show these exact sources; the code between ``# <<seg:NAME>>`` and ``# <<end>>`` is what a notebook cell prints.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent


# <<seg:rmsnorm>>
class RMSNorm(nn.Module):
    """x / rms(x) * weight  (no mean subtraction, no bias)."""

    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.w = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.w
# <<end>>


# <<seg:rope>>
def rope_inv_freq(dim: int, base: float = 10000.0) -> torch.Tensor:
    """theta_i = base^(-2i/dim): one rotation speed per pair of channels."""
    return 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))


class Rope(nn.Module):
    """Rotary position embedding on the last dim of x (shape ..., T, dim)."""

    def __init__(self, dim: int, inv_freq: torch.Tensor | None = None, mscale: float = 1.0):
        super().__init__()
        self.dim, self.mscale = dim, mscale
        self.register_buffer("inv_freq", rope_inv_freq(dim) if inv_freq is None else inv_freq.clone(), persistent=False)

    def angles(self, T: int, pos0: int = 0):
        pos = torch.arange(pos0, pos0 + T, device=self.inv_freq.device).float()
        ang = torch.outer(pos, self.inv_freq)                      # (T, dim/2)
        return ang.cos() * self.mscale, ang.sin() * self.mscale

    def apply(self, x, pos0: int = 0):
        cos, sin = self.angles(x.shape[-2], pos0)
        x1, x2 = x[..., : self.dim // 2], x[..., self.dim // 2:]
        return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], -1)
# <<end>>


# <<seg:yarn>>
def yarn_inv_freq(dim: int, scale: float, orig_T: int, alpha: float = 1.0, beta: float = 32.0, base: float = 10000.0):
    """YaRN: slow channels are stretched by `scale`, fast channels are left alone, a linear ramp in between.

    r_i = orig_T / wavelength_i = how many full turns channel i makes inside the original context.
        r < alpha  -> divide the frequency by `scale`  (position interpolation)
        r > beta   -> keep the frequency               (no change)
    """
    inv = rope_inv_freq(dim, base)
    r = orig_T / (2 * math.pi / inv)
    gamma = ((r - alpha) / (beta - alpha)).clamp(0, 1)         # 0 = interpolate, 1 = keep
    return (1 - gamma) * inv / scale + gamma * inv, gamma


def yarn_mscale(scale: float) -> float:
    """sqrt(1/t) = 0.1 ln(s) + 1: attention-temperature correction."""
    return 0.1 * math.log(scale) + 1.0
# <<end>>


# <<seg:mha>>
class MHA(nn.Module):
    """Standard multi-head attention with RoPE; caches K and V of every head."""

    def __init__(self, d: int, n_h: int, max_T: int = 4096):
        super().__init__()
        self.n_h, self.d_h = n_h, d // n_h
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.o = nn.Linear(d, d, bias=False)
        self.rope = Rope(self.d_h)

    def forward(self, h):
        B, T, d = h.shape
        q, k, v = self.qkv(h).view(B, T, 3, self.n_h, self.d_h).permute(2, 0, 3, 1, 4)
        q, k = self.rope.apply(q), self.rope.apply(k)
        o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.o(o.transpose(1, 2).reshape(B, T, d))
# <<end>>


# <<seg:mla>>
class MLA(nn.Module):
    """Multi-head Latent Attention (paper eqs. 1-11).

    Only c_kv (d_c numbers) and k_r (d_r numbers) per token have to be cached.
    """

    def __init__(self, d: int, n_h: int, d_h: int, d_c: int, d_cq: int, d_r: int):
        super().__init__()
        self.n_h, self.d_h, self.d_c, self.d_r = n_h, d_h, d_c, d_r
        # keys / values: compress with W_DKV, expand with W_UK / W_UV
        self.W_DKV = nn.Linear(d, d_c, bias=False)
        self.kv_norm = RMSNorm(d_c)
        self.W_UK = nn.Linear(d_c, n_h * d_h, bias=False)
        self.W_UV = nn.Linear(d_c, n_h * d_h, bias=False)
        self.W_KR = nn.Linear(d, d_r, bias=False)            # the single RoPE key shared by all heads
        # queries: also low-rank (saves activation memory in training)
        self.W_DQ = nn.Linear(d, d_cq, bias=False)
        self.q_norm = RMSNorm(d_cq)
        self.W_UQ = nn.Linear(d_cq, n_h * d_h, bias=False)
        self.W_QR = nn.Linear(d_cq, n_h * d_r, bias=False)
        self.W_O = nn.Linear(n_h * d_h, d, bias=False)
        self.rope = Rope(d_r)
        self.scale = (d_h + d_r) ** -0.5                     # 1/sqrt(d_h + d_r), eq. (10)

    def latents(self, h, pos0: int = 0):
        """What the cache stores for these tokens: c_kv (B,T,d_c) and the RoPE key k_r (B,T,d_r)."""
        c_kv = self.kv_norm(self.W_DKV(h))
        k_r = self.rope.apply(self.W_KR(h), pos0)
        return c_kv, k_r

    def queries(self, h, pos0: int = 0):
        B, T, _ = h.shape
        c_q = self.q_norm(self.W_DQ(h))
        q_c = self.W_UQ(c_q).view(B, T, self.n_h, self.d_h).transpose(1, 2)            # (B,h,T,d_h)
        q_r = self.rope.apply(self.W_QR(c_q).view(B, T, self.n_h, self.d_r).transpose(1, 2), pos0)
        return q_c, q_r

    def forward(self, h):
        """Training / prefill: expand the latent into per-head K and V and run ordinary attention."""
        B, T, _ = h.shape
        c_kv, k_r = self.latents(h)
        q_c, q_r = self.queries(h)
        k_c = self.W_UK(c_kv).view(B, T, self.n_h, self.d_h).transpose(1, 2)
        v = self.W_UV(c_kv).view(B, T, self.n_h, self.d_h).transpose(1, 2)
        q = torch.cat([q_c, q_r], -1)
        k = torch.cat([k_c, k_r.unsqueeze(1).expand(-1, self.n_h, -1, -1)], -1)
        o = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=self.scale)
        return self.W_O(o.transpose(1, 2).reshape(B, T, -1))

    def forward_absorbed(self, h, cache=None):
        """Inference: never build per-head K / V.  W_UK is folded into the query, W_UV into the output.

        cache = (c_kv, k_r) of the tokens already processed; returns (output, new_cache).
        """
        B, T, _ = h.shape
        pos0 = 0 if cache is None else cache[0].shape[1]
        c_new, kr_new = self.latents(h, pos0)
        c_kv = c_new if cache is None else torch.cat([cache[0], c_new], 1)
        k_r = kr_new if cache is None else torch.cat([cache[1], kr_new], 1)
        q_c, q_r = self.queries(h, pos0)
        W_UK = self.W_UK.weight.view(self.n_h, self.d_h, self.d_c)
        W_UV = self.W_UV.weight.view(self.n_h, self.d_h, self.d_c)
        q_abs = torch.einsum("bhtd,hdc->bhtc", q_c, W_UK)                               # query in latent space
        S = c_kv.shape[1]
        score = (q_abs @ c_kv.unsqueeze(1).transpose(-1, -2) + q_r @ k_r.unsqueeze(1).transpose(-1, -2)) * self.scale
        allowed = torch.arange(S, device=h.device)[None, :] <= (pos0 + torch.arange(T, device=h.device))[:, None]
        p = score.masked_fill(~allowed, float("-inf")).softmax(-1)                       # (B,h,T,S)
        o_lat = p @ c_kv.unsqueeze(1)                                                    # (B,h,T,d_c)
        o = torch.einsum("bhtc,hdc->bhtd", o_lat, W_UV)
        return self.W_O(o.transpose(1, 2).reshape(B, T, -1)), (c_kv, k_r)
# <<end>>


def kv_cache_per_token(kind: str, n_h=128, d_h=128, d_c=512, d_r=64, n_kv=8) -> int:
    """Numbers cached per token per layer."""
    return {"mha": 2 * n_h * d_h, "gqa": 2 * n_kv * d_h, "mqa": 2 * d_h, "mla": d_c + d_r}[kind]


# <<seg:expert>>
class Expert(nn.Module):
    """One FFN expert (SwiGLU): down( silu(gate(x)) * up(x) )."""

    def __init__(self, d: int, d_ff: int):
        super().__init__()
        self.gate = nn.Linear(d, d_ff, bias=False)
        self.up = nn.Linear(d, d_ff, bias=False)
        self.down = nn.Linear(d_ff, d, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))
# <<end>>


# <<seg:moe>>
class DeepSeekMoE(nn.Module):
    """Shared experts + fine-grained routed experts (paper eqs. 12-16).

    n_groups / topk_groups implement node-limited routing: experts are split into n_groups "nodes";
    every token may only use experts from its topk_groups best nodes.
    """

    def __init__(self, d, d_ff, n_shared, n_routed, top_k, n_groups=1, topk_groups=1):
        super().__init__()
        self.top_k, self.n_routed, self.n_groups, self.topk_groups = top_k, n_routed, n_groups, topk_groups
        self.shared = nn.ModuleList(Expert(d, d_ff) for _ in range(n_shared))
        self.routed = nn.ModuleList(Expert(d, d_ff) for _ in range(n_routed))
        self.centroids = nn.Parameter(torch.randn(n_routed, d) * d ** -0.5)          # e_i in eq. (15)
        self.register_buffer("bias", torch.zeros(n_routed))                           # b_i in eq. (16): routing only
        self.last = {}

    def route(self, u):
        aff = torch.sigmoid(u @ self.centroids.t())                                   # s_{i,t}, eq. (15)
        sel = aff.detach() + self.bias                                                # what top-k looks at, eq. (16)
        if self.n_groups > 1:                                                         # node-limited routing
            N = u.shape[0]
            per = self.n_routed // self.n_groups
            k_in = self.top_k // self.topk_groups
            group_score = sel.view(N, self.n_groups, per).topk(k_in, dim=-1).values.sum(-1)
            keep = group_score.topk(self.topk_groups, dim=-1).indices
            ok = torch.zeros(N, self.n_groups, dtype=torch.bool, device=u.device).scatter(1, keep, True)
            sel = sel.masked_fill(~ok.repeat_interleave(per, dim=1), float("-inf"))
        idx = sel.topk(self.top_k, dim=-1).indices                                    # chosen experts
        g = aff.gather(1, idx)                                                        # gate = ORIGINAL affinity (no bias)
        g = g / g.sum(-1, keepdim=True)                                               # eq. (13)
        return aff, idx, g

    def forward(self, x):
        B, T, d = x.shape
        u = x.reshape(-1, d)
        aff, idx, g = self.route(u)
        out = torch.zeros_like(u)
        for sh in self.shared:                                                        # always-on experts
            out = out + sh(u)
        for e in range(self.n_routed):                                                # only the chosen tokens
            tok, slot = (idx == e).nonzero(as_tuple=True)
            if tok.numel():
                out.index_add_(0, tok, g[tok, slot].unsqueeze(-1) * self.routed[e](u[tok]))
        self.last = dict(aff=aff, idx=idx, load=torch.bincount(idx.reshape(-1), minlength=self.n_routed))
        return out.view(B, T, d)
# <<end>>


# <<seg:balance>>
@torch.no_grad()
def update_bias(moe: DeepSeekMoE, gamma: float):
    """Auxiliary-loss-free balancing: overloaded expert -> bias down by gamma, underloaded -> up by gamma."""
    load = moe.last["load"].float()
    moe.bias += gamma * torch.sign(load.mean() - load)


def seq_balance_loss(moe: DeepSeekMoE, B: int, T: int, alpha: float):
    """Complementary sequence-wise balance loss, eqs. (17)-(20): alpha * sum_i f_i P_i per sequence."""
    aff, idx = moe.last["aff"], moe.last["idx"]
    Nr, K = moe.n_routed, idx.shape[1]
    s_norm = aff.view(B, T, Nr) / aff.view(B, T, Nr).sum(-1, keepdim=True)            # s'_{i,t}
    P = s_norm.mean(1)                                                                # (B, Nr)
    picked = F.one_hot(idx.view(B, T, K), Nr).sum(2).float()                          # (B, T, Nr)
    f = picked.mean(1) * Nr / K                                                       # f_i = Nr/(K T) sum_t 1(...)
    return alpha * (f * P).sum(-1).mean()
# <<end>>


# ------------------------------------------------------------------ a tiny LM built from the blocks
@dataclass
class LMConfig:
    vocab: int = 128
    d: int = 128
    n_layers: int = 4
    n_heads: int = 4
    d_ff: int = 256
    attn: str = "mha"            # 'mha' | 'mla'
    ffn: str = "dense"           # 'dense' | 'moe'
    first_dense: int = 1         # first layers keep a dense FFN (as in the paper: first 3 of 61)
    # MLA
    d_c: int = 32
    d_cq: int = 48
    d_r: int = 16
    # MoE
    n_shared: int = 1
    n_routed: int = 8
    top_k: int = 2
    moe_ff: int = 64
    n_groups: int = 1
    topk_groups: int = 1
    # MTP
    mtp_depth: int = 0
    mtp_lambda: float = 0.3


class DenseFFN(Expert):
    pass


class Block(nn.Module):
    def __init__(self, d, attn, ffn):
        super().__init__()
        self.n1, self.attn, self.n2, self.ffn = RMSNorm(d), attn, RMSNorm(d), ffn

    def forward(self, x):
        x = x + self.attn(self.n1(x))
        return x + self.ffn(self.n2(x))


def make_block(cfg: LMConfig, layer: int) -> Block:
    d_h = cfg.d // cfg.n_heads
    attn = MHA(cfg.d, cfg.n_heads) if cfg.attn == "mha" else MLA(cfg.d, cfg.n_heads, d_h, cfg.d_c, cfg.d_cq, cfg.d_r)
    use_moe = cfg.ffn == "moe" and layer >= cfg.first_dense
    ffn = DeepSeekMoE(cfg.d, cfg.moe_ff, cfg.n_shared, cfg.n_routed, cfg.top_k, cfg.n_groups, cfg.topk_groups) \
        if use_moe else DenseFFN(cfg.d, cfg.d_ff)
    return Block(cfg.d, attn, ffn)


# <<seg:mtp>>
class MTPModule(nn.Module):
    """One multi-token-prediction module (paper eqs. 21-22): RMSNorm both inputs, concat, project 2d->d, one block."""

    def __init__(self, cfg: LMConfig):
        super().__init__()
        self.n_h, self.n_e = RMSNorm(cfg.d), RMSNorm(cfg.d)
        self.proj = nn.Linear(2 * cfg.d, cfg.d, bias=False)                           # M_k in R^{d x 2d}
        self.block = make_block(cfg, layer=0)                                         # TRM_k (dense FFN here)

    def forward(self, h_prev, emb_future):
        return self.block(self.proj(torch.cat([self.n_h(h_prev), self.n_e(emb_future)], -1)))
# <<end>>


class TinyLM(nn.Module):
    def __init__(self, cfg: LMConfig):
        super().__init__()
        self.cfg = cfg
        self.emb = nn.Embedding(cfg.vocab, cfg.d)
        self.blocks = nn.ModuleList(make_block(cfg, i) for i in range(cfg.n_layers))
        self.norm = RMSNorm(cfg.d)
        self.mtp = nn.ModuleList(MTPModule(cfg) for _ in range(cfg.mtp_depth))
        nn.init.normal_(self.emb.weight, std=0.02)
        for n, p in self.named_parameters():
            if p.dim() == 2 and "emb" not in n and "centroids" not in n:
                nn.init.normal_(p, std=0.02)

    def head(self, h):                                   # output head shared with every MTP module (weights tied to emb)
        return self.norm(h) @ self.emb.weight.t()

    def trunk(self, idx):
        h = self.emb(idx)
        for b in self.blocks:
            h = b(h)
        return h

    def forward(self, idx):
        return self.head(self.trunk(idx))

    def moes(self):
        return [b.ffn for b in self.blocks if isinstance(b.ffn, DeepSeekMoE)]


# <<seg:mtp_loss>>
def mtp_forward(model: TinyLM, seq, h=None):
    """Run the MTP chain.  seq (B, T+1) ids; h = trunk output on seq[:, :-1].

    Depth k works on positions i = 0 .. T-k-1: it combines the previous depth's hidden state at i with the
    embedding of token i+k (the token that is *already known* when we predict token i+k+1), eqs. (21)-(23).
    Returns [(logits_k, targets_k), ...].
    """
    T = seq.shape[1] - 1
    h_prev = model.trunk(seq[:, :-1]) if h is None else h
    out = []
    for k, mod in enumerate(model.mtp, start=1):
        n = T - k                                                                    # each depth is one position shorter
        h_k = mod(h_prev[:, :n], model.emb(seq[:, k:k + n]))                          # shared embedding
        out.append((model.head(h_k), seq[:, k + 1:k + 1 + n]))                        # shared output head
        h_prev = h_k
    return out


def lm_loss(model: TinyLM, seq, seq_balance_alpha: float = 0.0, batch_balance_alpha: float = 0.0):
    """Total loss = L_main + lambda/D * sum_k L^k_MTP (+ optional sequence-wise balance loss)."""
    cfg = model.cfg
    h = model.trunk(seq[:, :-1])
    main = F.cross_entropy(model.head(h).reshape(-1, cfg.vocab), seq[:, 1:].reshape(-1))
    parts, total = {"main": main.detach()}, main
    mtp = [F.cross_entropy(lg.reshape(-1, cfg.vocab), tg.reshape(-1)) for lg, tg in mtp_forward(model, seq, h)]
    for k, l in enumerate(mtp, start=1):
        parts[f"mtp{k}"] = l.detach()
    if mtp:
        total = total + cfg.mtp_lambda * torch.stack(mtp).mean()                      # eq. (25)
    if seq_balance_alpha > 0:
        bal = sum(seq_balance_loss(m, seq.shape[0], seq.shape[1] - 1, seq_balance_alpha) for m in model.moes())
        parts["seq_bal"] = bal.detach()
        total = total + bal
    if batch_balance_alpha > 0:                                                       # same loss, but over the whole batch
        bal = sum(seq_balance_loss(m, 1, seq.shape[0] * (seq.shape[1] - 1), batch_balance_alpha) for m in model.moes())
        parts["batch_bal"] = bal.detach()
        total = total + bal
    return total, parts
# <<end>>


# ------------------------------------------------------------------ data + training loop
def load_corpus() -> str:
    return (HERE / "corpus.txt").read_text(encoding="utf-8")


class CharData:
    def __init__(self, text: str | None = None, val_frac: float = 0.1):
        text = load_corpus() if text is None else text
        chars = sorted(set(text))
        self.stoi = {c: i for i, c in enumerate(chars)}
        self.itos = chars
        ids = torch.tensor([self.stoi[c] for c in text], dtype=torch.long)
        n = int(len(ids) * (1 - val_frac))
        self.train, self.val = ids[:n], ids[n:]
        self.vocab = len(chars)

    def batch(self, split: str, B: int, T: int, gen: torch.Generator, device=None):
        data = self.train if split == "train" else self.val
        ix = torch.randint(0, len(data) - T - 1, (B,), generator=gen)
        out = torch.stack([data[i:i + T + 1] for i in ix])
        return out if device is None else out.to(device)

    def decode(self, ids):
        return "".join(self.itos[int(i)] for i in ids)


@torch.no_grad()
def evaluate(model: TinyLM, data: CharData, T: int, n_batches: int = 20, B: int = 16, seed: int = 123):
    model.eval()
    dev = next(model.parameters()).device
    gen = torch.Generator().manual_seed(seed)
    tot = {}
    for _ in range(n_batches):
        _, parts = lm_loss(model, data.batch("val", B, T, gen, dev))
        for k, v in parts.items():
            tot[k] = tot.get(k, 0.0) + float(v) / n_batches
    model.train()
    return tot


def train_lm(cfg: LMConfig, data: CharData, steps=300, B=16, T=64, lr=3e-3, seed=0, gamma=0.0,
             seq_alpha=0.0, batch_alpha=0.0, log_every=50, record_load=False, quiet=True, device="cpu"):
    """Train a TinyLM.  gamma>0 turns on auxiliary-loss-free bias updates after every step."""
    torch.manual_seed(seed)
    model = TinyLM(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.1)
    gen = torch.Generator().manual_seed(seed + 1)
    hist = dict(step=[], loss=[], main=[], load=[])
    for step in range(1, steps + 1):
        cur = lr * min(1.0, step / 20) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * step / steps)))
        for g_ in opt.param_groups:
            g_["lr"] = cur
        loss, parts = lm_loss(model, data.batch("train", B, T, gen, device), seq_alpha, batch_alpha)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        for m in model.moes():
            if gamma > 0:
                update_bias(m, gamma)
        if record_load and model.moes():
            hist["load"].append(torch.stack([m.last["load"].float() for m in model.moes()]))
        if step % log_every == 0 or step == steps:
            hist["step"].append(step)
            hist["loss"].append(float(loss.detach()))
            hist["main"].append(float(parts["main"]))
            if not quiet:
                print(f"step {step:4d}  loss {float(loss):.3f}  main {float(parts['main']):.3f}")
    return model, hist


# ------------------------------------------------------------------ FP8 (E4M3) with fine-grained scaling
FP8_MAX = 448.0

# <<seg:fp8_quant>>
def quant_tile(x, tile: int = 128):
    """Activations: one scale per 1 x `tile` group along the last (contraction) dim.  x: (M, K), K % tile == 0."""
    M, K = x.shape
    g = x.reshape(M, K // tile, tile)
    s = (g.abs().amax(-1) / FP8_MAX).clamp_min(1e-12)                    # (M, K/tile)
    q = (g / s.unsqueeze(-1)).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)   # values now fit E4M3
    return q.reshape(M, K), s


def quant_block(w, blk: int = 128):
    """Weights: one scale per blk x blk block.  w: (N, K).  Returns fp8 (N,K) and scales (N/blk, K/blk)."""
    N, K = w.shape
    g = w.reshape(N // blk, blk, K // blk, blk)
    s = (g.abs().amax(dim=(1, 3)) / FP8_MAX).clamp_min(1e-12)            # (N/blk, K/blk)
    q = (g / s[:, None, :, None]).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    return q.reshape(N, K), s


def quant_tensor(x):
    """The 'standard practice' baseline: ONE scale for the whole tensor."""
    s = (x.abs().max() / FP8_MAX).clamp_min(1e-12)
    return (x / s).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn), s


def dequant_tile(q, s, tile: int = 128):
    M, K = q.shape
    return (q.float().reshape(M, K // tile, tile) * s.unsqueeze(-1)).reshape(M, K)


def dequant_block(q, s, blk: int = 128):
    N, K = q.shape
    return (q.float().reshape(N // blk, blk, K // blk, blk) * s[:, None, :, None]).reshape(N, K)
# <<end>>


# <<seg:fp8_gemm>>
def round_sig_bits(x, bits: int):
    """Keep only `bits` significant bits (emulates a narrow accumulator register)."""
    m, e = torch.frexp(x)
    return torch.ldexp(torch.round(m * 2.0 ** bits) / 2.0 ** bits, e)


def accumulate_limited(a, b, interval: int, mma_k: int = 32, acc_bits: int | None = 14):
    """a @ b.T where a Tensor-Core-like accumulator keeps only `acc_bits` significant bits.

    The K axis is cut into MMA steps of `mma_k`.  After every `interval` elements the accumulator is
    promoted: added to an FP32 running sum and cleared.  acc_bits=None -> exact FP32 everywhere.
    """
    M, K = a.shape
    total = torch.zeros(M, b.shape[0])
    for k0 in range(0, K, interval):
        acc = torch.zeros(M, b.shape[0])
        for k1 in range(k0, min(k0 + interval, K), mma_k):                              # one MMA instruction
            part = a[:, k1:k1 + mma_k] @ b[:, k1:k1 + mma_k].t()
            acc = acc + part if acc_bits is None else round_sig_bits(acc + part, acc_bits)
        total = total + acc                                                             # promotion to FP32
    return total


def fp8_gemm(aq, sa, bq, sb, interval: int = 128, mma_k: int = 32, acc_bits: int | None = 14, group: int = 128):
    """C = A @ B^T from FP8 operands with per-group scales along K  (paper Fig. 7).

    aq (M,K) fp8, sa (M,K/group)    bq (N,K) fp8, sb (N,K/group) (block scales already expanded per row).
    Each scale group is accumulated on the narrow accumulator, then multiplied by both scales and added to an FP32 sum.
    """
    assert group % interval == 0, "the promotion interval must divide the scale group"
    a, b = aq.float(), bq.float()
    out = torch.zeros(a.shape[0], b.shape[0])
    for gi in range(a.shape[1] // group):
        sl = slice(gi * group, (gi + 1) * group)
        part = accumulate_limited(a[:, sl], b[:, sl], interval, mma_k, acc_bits)
        out = out + part * sa[:, gi:gi + 1] * sb[:, gi].unsqueeze(0)                    # dequantise on "CUDA cores", FP32
    return out
# <<end>>


def _pad_to(x, mult, dim):
    r = (-x.shape[dim]) % mult
    if r == 0:
        return x
    pad = [0, 0] * x.dim()
    pad[2 * (x.dim() - 1 - dim) + 1] = r
    return F.pad(x, pad)


def fp8_matmul_nt(a, b, *, b_is_weight_block: bool, acc_bits: int | None = 14, interval: int = 128):
    """a (M,K) activation-like, b (N,K).  Returns a @ b.T with FP8 inputs (1x128 tiles on a, 128x128 blocks or 1x128 tiles on b)."""
    a, b = _pad_to(a, 128, 1), _pad_to(b, 128, 1)
    aq, sa = quant_tile(a)
    if b_is_weight_block:
        b = _pad_to(b, 128, 0)
        bq, sb = quant_block(b)
        sb = sb.repeat_interleave(128, dim=0)
    else:
        bq, sb = quant_tile(b)
    return fp8_gemm(aq, sa, bq, sb, interval=interval, acc_bits=acc_bits)


# <<seg:fp8_linear>>
class _FP8LinearFn(torch.autograd.Function):
    """Linear layer whose three GEMMs (Fprop, Dgrad, Wgrad) all run on E4M3 operands (paper Fig. 6)."""

    @staticmethod
    def forward(ctx, x, w, acc_bits):
        ctx.save_for_backward(x, w)
        ctx.acc_bits = acc_bits
        # Fprop: Y = X W^T      contraction over K=in_features: X in 1x128 tiles, W in 128x128 blocks
        return fp8_matmul_nt(x, w, b_is_weight_block=True, acc_bits=acc_bits)[:, : w.shape[0]]

    @staticmethod
    def backward(ctx, dy):
        x, w = ctx.saved_tensors
        # Dgrad: dX = dY W      contraction over N=out_features: dY in 1x128 tiles, W^T in 128x128 blocks
        dx = fp8_matmul_nt(dy, w.t().contiguous(), b_is_weight_block=True, acc_bits=ctx.acc_bits)[:, : x.shape[1]]
        # Wgrad: dW = dY^T X    contraction over M=tokens: both operands in 1x128 tiles along the token axis
        dw = fp8_matmul_nt(dy.t().contiguous(), x.t().contiguous(), b_is_weight_block=False, acc_bits=ctx.acc_bits)
        return dx, dw[: w.shape[0], : w.shape[1]], None


class FP8Linear(nn.Module):
    """Master weights stay FP32; every matmul sees E4M3 tensors with fine-grained scales."""

    def __init__(self, d_in, d_out, acc_bits: int | None = 14):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(d_out, d_in) * d_in ** -0.5)
        self.acc_bits = acc_bits

    def forward(self, x):
        lead = x.shape[:-1]
        y = _FP8LinearFn.apply(x.reshape(-1, x.shape[-1]), self.weight, self.acc_bits)
        return y.reshape(*lead, -1)
# <<end>>
