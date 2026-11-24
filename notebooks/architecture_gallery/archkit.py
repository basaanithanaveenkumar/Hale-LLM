"""archkit: from-scratch building blocks for the 'How LLMs build on each other' notebook series.

Each `# <<seg:name>> ... # <<end>>` block is shown inside a notebook cell (the notebook builder
pulls the source out of this file), so what the reader sees is exactly what the tests run.

Contents
  common      RMSNorm, rotary embeddings, explicit softmax attention
  attn        one attention layer with GQA/MQA, output gate, learned sink logit, sliding window,
              StreamingLLM cache positions, and an external (sparse) mask
  dsa         DeepSeek-Sparse-Attention lightning indexer, top-k mask, indexer KL loss
  delta       linear attention / delta rule / gated delta rule (recurrent + chunkwise), KDA
  moe         standard MoE and LatentMoE, quantile balancing
  attnres     Full / Block attention residuals
  lm          MiniLM (pluggable attention, FFN and residual stack), data, training
"""
from __future__ import annotations
import math
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

# <<seg:common>>
class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.w = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.w


def rope_angles(pos, dh, base=10000.0):
    """pos: integer tensor of any shape -> angles [..., dh/2]."""
    inv = base ** (-torch.arange(0, dh, 2, dtype=torch.float32) / dh)
    return pos.to(torch.float32)[..., None] * inv


def apply_rope(x, pos):
    """x: [..., dh] (pairs are (even, odd) channels); pos broadcasts against x[..., 0]."""
    ang = rope_angles(pos, x.shape[-1])
    c, s = ang.cos(), ang.sin()
    x1, x2 = x[..., 0::2], x[..., 1::2]
    out = torch.empty_like(x)
    out[..., 0::2] = x1 * c - x2 * s
    out[..., 1::2] = x1 * s + x2 * c
    return out


def attend(q, k, v, mask, sink=None):
    """Explicit softmax attention.
    q [B,H,T,d]  k,v [B,H,S,d]  mask bool, broadcastable to [B,H,T,S] (True = may look)
    sink: optional [H] learned logit. It is one extra softmax column that has no value vector:
          the probability it takes is simply thrown away (gpt-oss style).
    returns out [B,H,T,d], probs [B,H,T,S] (without the sink column)."""
    B, H, T, d = q.shape
    s = (q @ k.transpose(-1, -2)) / math.sqrt(d)
    s = s.masked_fill(~mask, float("-inf"))
    if sink is not None:
        s = torch.cat([s, sink.view(1, H, 1, 1).expand(B, H, T, 1)], dim=-1)
    p = s.softmax(-1)
    if sink is not None:
        p = p[..., :-1]
    return p @ v, p
# <<end>>


# <<seg:attn>>
class Attn(nn.Module):
    """Causal self-attention with the knobs this series studies.

    n_kv      number of key/value heads (n_kv == n_heads is MHA, 1 is MQA, in between is GQA)
    gate      None | "head" | "elem": sigmoid gate on the attention output, computed from the layer input
    sink      True: one learned sink logit per head
    """
    def __init__(self, d, n_heads, n_kv=None, gate=None, sink=False, rope=True):
        super().__init__()
        n_kv = n_kv or n_heads
        assert n_heads % n_kv == 0
        self.d, self.H, self.Hkv, self.dh = d, n_heads, n_kv, d // n_heads
        self.wq = nn.Linear(d, n_heads * self.dh, bias=False)
        self.wk = nn.Linear(d, n_kv * self.dh, bias=False)
        self.wv = nn.Linear(d, n_kv * self.dh, bias=False)
        self.wo = nn.Linear(n_heads * self.dh, d, bias=False)
        self.gate = gate
        if gate == "elem":
            self.wg = nn.Linear(d, n_heads * self.dh, bias=False)
        elif gate == "head":
            self.wg = nn.Linear(d, n_heads, bias=False)
        self.sink = nn.Parameter(torch.zeros(n_heads)) if sink else None
        self.rope = rope
        self.keep = False                    # set True to store the attention probabilities in self.p
        self.p = None
        self.g = None

    def split(self, x, n):
        B, T, _ = x.shape
        return x.view(B, T, n, self.dh).transpose(1, 2)

    def forward(self, x, window=None, n_sink_tokens=0, cache_pos=False, extra_mask=None):
        """window: keep only the last `window` tokens (sliding-window attention).
        n_sink_tokens: in addition keep the first n tokens forever (StreamingLLM).
        cache_pos: give RoPE the position *inside the cache* rather than the original position.
        extra_mask: bool [B,T,T] (e.g. from DSA's top-k) ANDed onto the causal mask."""
        B, T, _ = x.shape
        q, k, v = self.split(self.wq(x), self.H), self.split(self.wk(x), self.Hkv), self.split(self.wv(x), self.Hkv)
        g = self.H // self.Hkv
        pos = torch.arange(T)
        if window is not None and cache_pos:
            out, p = self._streaming(q, k, v, window, n_sink_tokens)
        else:
            if self.rope:
                q, k = apply_rope(q, pos[None, None, :]), apply_rope(k, pos[None, None, :])
            k, v = k.repeat_interleave(g, 1), v.repeat_interleave(g, 1)
            i, j = pos[:, None], pos[None, :]
            mask = j <= i
            if window is not None:
                mask = mask & ((i - j < window) | (j < n_sink_tokens))
            mask = mask[None, None]
            if extra_mask is not None:
                mask = mask & extra_mask[:, None]
            out, p = attend(q, k, v, mask, self.sink)
        if self.keep:
            self.p = p.detach()
        out = out.transpose(1, 2)                                   # [B,T,H,dh]
        if self.gate is not None:
            gs = torch.sigmoid(self.wg(x))
            self.g = gs.detach() if self.keep else None
            out = out * (gs.view(B, T, self.H, 1) if self.gate == "head" else gs.view(B, T, self.H, self.dh))
        return self.wo(out.reshape(B, T, self.H * self.dh))

    def _streaming(self, q, k, v, window, n_sink):
        """StreamingLLM: query t sees n_sink first tokens + the last `window` tokens, and RoPE uses the
        slot index inside that cache, so the model never sees a distance larger than n_sink+window."""
        B, H, T, dh = q.shape
        S = n_sink + window
        t = torch.arange(T)[:, None]                               # [T,1]
        slot = torch.arange(S)[None, :]                            # [1,S]
        n_cached = torch.clamp(t + 1, max=S)                       # entries in the cache when t is decoded
        # before eviction starts the cache holds tokens 0..t; afterwards sinks + the last `window` tokens
        evicting = (t + 1) > S
        tok = torch.where(evicting & (slot >= n_sink), t - (S - 1 - slot), slot)      # token index held in each slot
        valid = slot < n_cached
        tok = tok.clamp(0, T - 1)
        kg = k[:, :, tok]                                          # [B,Hkv,T,S,dh]
        vg = v[:, :, tok]
        g = H // self.Hkv
        kg, vg = kg.repeat_interleave(g, 1), vg.repeat_interleave(g, 1)
        qpos = (n_cached - 1)[None, None, :, 0]                    # [1,1,T]
        if self.rope:
            q = apply_rope(q, qpos)
            kg = apply_rope(kg, slot[None, None, :, :].expand(1, 1, T, S))
        s = (q[:, :, :, None, :] * kg).sum(-1) / math.sqrt(dh)     # [B,H,T,S]
        s = s.masked_fill(~valid[None, None], float("-inf"))
        if self.sink is not None:
            s = torch.cat([s, self.sink.view(1, H, 1, 1).expand(B, H, T, 1)], -1)
        p = s.softmax(-1)
        if self.sink is not None:
            p = p[..., :-1]
        out = (p[..., None] * vg).sum(3)
        return out, p
# <<end>>


# <<seg:dsa>>
class LightningIndexer(nn.Module):
    """DeepSeek Sparse Attention, step 1.  I[t,s] = sum_j w[t,j] * ReLU(q_j[t] . k[s])   (paper, eq. 1)
    A few tiny heads (h_i of width d_i) score every earlier token cheaply."""
    def __init__(self, d, h_i=4, d_i=16):
        super().__init__()
        self.h_i, self.d_i = h_i, d_i
        self.wq = nn.Linear(d, h_i * d_i, bias=False)
        self.wk = nn.Linear(d, d_i, bias=False)
        self.ww = nn.Linear(d, h_i, bias=False)

    def forward(self, x):
        B, T, _ = x.shape
        q = self.wq(x).view(B, T, self.h_i, self.d_i)
        k = self.wk(x)                                              # one key per token, shared by indexer heads
        pos = torch.arange(T)                                       # the indexer needs positions too (the paper rotates part of q and k);
        q = apply_rope(q.transpose(1, 2), pos[None, None, :]).transpose(1, 2)    # here we rotate all d_i channels
        k = apply_rope(k, pos[None, :])
        w = self.ww(x)                                              # [B,T,h_i]
        s = torch.einsum("bthd,bsd->bths", q, k).relu()
        return torch.einsum("bth,bths->bts", w, s)                  # [B,T,S]


def topk_mask(I, k):
    """DSA step 2: keep only the k highest-scoring earlier tokens per query (causal)."""
    B, T, S = I.shape
    causal = torch.tril(torch.ones(T, S, dtype=torch.bool))
    I = I.masked_fill(~causal, float("-inf"))
    kk = min(k, S)
    idx = I.topk(kk, dim=-1).indices
    m = torch.zeros(B, T, S, dtype=torch.bool).scatter_(-1, idx, True)
    return m & causal


def indexer_kl(I, p_dense, mask=None):
    """Indexer training loss (paper eq. 3/4): KL( p_dense || softmax(I) ) summed over queries.
    p_dense [B,T,S] is the main attention summed over heads and L1-normalised.
    With `mask` (sparse stage) both distributions are restricted to the selected tokens."""
    B, T, S = I.shape
    allow = torch.tril(torch.ones(T, S, dtype=torch.bool))[None].expand(B, T, S)
    if mask is not None:
        allow = allow & mask
    logq = I.masked_fill(~allow, float("-inf")).log_softmax(-1)
    p = p_dense * allow
    p = p / p.sum(-1, keepdim=True).clamp_min(1e-9)
    kl = torch.where(p > 0, p * (p.clamp_min(1e-12).log() - logq), torch.zeros_like(p))
    return kl.sum(-1).mean()
# <<end>>


# <<seg:delta>>
def linear_attn_recurrent(q, k, v, alpha=None, beta=None, rule="delta"):
    """One head, recurrent form.  q,k [T,dk]  v [T,dv]  alpha,beta [T]  ->  o [T,dv], S [dv,dk].
    rule = "linear"  : S_t = S_{t-1} + v k^T                                  (plain linear attention)
    rule = "decay"   : S_t = a S_{t-1} + v k^T                                (Mamba-2 style forget gate)
    rule = "delta"   : S_t = S_{t-1}(I - b k k^T) + b v k^T                   (DeltaNet)
    rule = "gdelta"  : S_t = S_{t-1} a (I - b k k^T) + b v k^T                (Gated DeltaNet, paper eq. 10)"""
    T, dk = k.shape
    dv = v.shape[1]
    S = torch.zeros(dv, dk, dtype=q.dtype)
    o = []
    for t in range(T):
        kt, vt = k[t], v[t]
        a = alpha[t] if alpha is not None else 1.0
        b = beta[t] if beta is not None else 1.0
        if rule == "linear":
            S = S + torch.outer(vt, kt)
        elif rule == "decay":
            S = a * S + torch.outer(vt, kt)
        elif rule == "delta":
            S = S - b * torch.outer(S @ kt, kt) + b * torch.outer(vt, kt)
        elif rule == "gdelta":
            S = a * (S - b * torch.outer(S @ kt, kt)) + b * torch.outer(vt, kt)
        o.append(S @ q[t])
    return torch.stack(o), S


def gated_delta_chunkwise(q, k, v, alpha, beta, C=8):
    """Gated DeltaNet, chunkwise form (paper sec. 3.3): chunks are processed with matmuls and one small
    triangular solve instead of C sequential rank-1 updates.  Returns o [T,dv], final S [dv,dk]."""
    T, dk = k.shape
    dv = v.shape[1]
    S = torch.zeros(dv, dk, dtype=q.dtype)
    outs = []
    for s0 in range(0, T, C):
        Q, Kc, V = q[s0:s0 + C], k[s0:s0 + C], v[s0:s0 + C]
        a, b = alpha[s0:s0 + C], beta[s0:s0 + C]
        n = Q.shape[0]
        g = torch.cumprod(a, 0)                                     # gamma_r: decay from chunk start to r
        Gam = g[:, None] / g[None, :]                               # Gamma_ij = gamma_i / gamma_j
        low = torch.tril(torch.ones(n, n, dtype=q.dtype))
        A = torch.tril(b[:, None] * (Gam * (Kc @ Kc.T)), -1)        # strictLower(diag(b) (Gamma o K K^T))
        Tm = torch.linalg.solve(torch.eye(n, dtype=q.dtype) + A, torch.diag(b))    # (I + A)^-1 diag(b)
        U = Tm @ V                                                  # "pseudo values" (decay already inside)
        Wm = Tm @ (g[:, None] * Kc)                                 # decayed "pseudo keys"
        Ucorr = U - Wm @ S.T                                        # what each token writes after removing what S already predicts
        o = (g[:, None] * Q) @ S.T + (Gam * (Q @ Kc.T) * low) @ Ucorr
        outs.append(o)
        S = g[-1] * S + Ucorr.T @ ((g[-1] / g)[:, None] * Kc)
    return torch.cat(outs), S


def kda_recurrent(q, k, v, alpha, beta):
    """Kimi Delta Attention (Kimi Linear paper eq. 1).  Same as the gated delta rule, but the decay
    alpha_t is a VECTOR over the key channels.  Layout follows the paper: S is [dk, dv].
        S_t = (I - b k k^T) Diag(alpha_t) S_{t-1} + b k v^T ;   o_t = S_t^T q_t"""
    T, dk = k.shape
    dv = v.shape[1]
    S = torch.zeros(dk, dv, dtype=q.dtype)
    o = []
    for t in range(T):
        kt, vt, at, bt = k[t], v[t], alpha[t], beta[t]
        S = at[:, None] * S
        S = S - bt * torch.outer(kt, kt @ S) + bt * torch.outer(kt, vt)
        o.append(S.T @ q[t])
    return torch.stack(o), S


def kda_chunkwise(q, k, v, alpha, beta, C=8):
    """KDA, chunkwise (Kimi Linear sec. 3.1).  alpha [T,dk] per-channel decay.
    The trick that lets per-channel decay stay a matmul: fold the cumulative decay into q and k."""
    T, dk = k.shape
    dv = v.shape[1]
    S = torch.zeros(dk, dv, dtype=q.dtype)
    outs = []
    for s0 in range(0, T, C):
        Q, Kc, V = q[s0:s0 + C], k[s0:s0 + C], v[s0:s0 + C]
        a, b = alpha[s0:s0 + C], beta[s0:s0 + C]
        n = Q.shape[0]
        g = torch.cumprod(a, 0)                                     # [n,dk]: per-channel decay from chunk start
        Kd = g * Kc                                                 # k decayed forward to the chunk start frame
        Ki = Kc / g                                                 # k "undecayed" (used so ratios g_i/g_j appear)
        low = torch.tril(torch.ones(n, n, dtype=q.dtype))
        A = torch.tril(b[:, None] * (Kd @ Ki.T), -1)
        Tm = torch.linalg.solve(torch.eye(n, dtype=q.dtype) + A, torch.diag(b))
        W, U = Tm @ Kd, Tm @ V
        Ucorr = U - W @ S
        o = (g * Q) @ S + ((g * Q) @ Ki.T * low) @ Ucorr
        outs.append(o)
        S = g[-1][:, None] * S + (g[-1] / g * Kc).T @ Ucorr
    return torch.cat(outs), S
# <<end>>


# <<seg:moe>>
class SwiGLU(nn.Module):
    def __init__(self, d, f):
        super().__init__()
        self.w1, self.w3, self.w2 = nn.Linear(d, f, bias=False), nn.Linear(d, f, bias=False), nn.Linear(f, d, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class MoE(nn.Module):
    """Routed mixture of experts.
    latent=None : ordinary MoE, experts live in the model width d.
    latent=l    : LatentMoE.  The router still reads the full d-wide token, but the token is first projected
                  d -> l, ALL experts run in width l, and the mixed result is projected l -> d.
    bias        : per-expert selection bias (used for choosing experts only, never for the gate value)."""
    def __init__(self, d, n_experts, topk, f, latent=None, n_shared=0, f_shared=None):
        super().__init__()
        self.d, self.E, self.k = d, n_experts, topk
        w = latent or d
        self.latent = latent
        if latent:
            self.down, self.up = nn.Linear(d, latent, bias=False), nn.Linear(latent, d, bias=False)
        self.router = nn.Linear(d, n_experts, bias=False)
        self.experts = nn.ModuleList([SwiGLU(w, f) for _ in range(n_experts)])
        self.shared = SwiGLU(d, f_shared or f) if n_shared else None
        self.register_buffer("bias", torch.zeros(n_experts))
        self.load = torch.zeros(n_experts)

    def route(self, x, bias=None):
        aff = torch.sigmoid(self.router(x))                         # [N,E]
        sel = (aff + (self.bias if bias is None else bias)).topk(self.k, -1).indices
        g = aff.gather(-1, sel)
        return sel, g / g.sum(-1, keepdim=True)

    def forward(self, x):
        shp = x.shape
        x = x.reshape(-1, shp[-1])
        sel, gate = self.route(x)
        z = self.down(x) if self.latent else x
        out = torch.zeros_like(z)
        for e in range(self.E):
            tok, slot = (sel == e).nonzero(as_tuple=True)
            if tok.numel():
                out.index_add_(0, tok, gate[tok, slot, None] * self.experts[e](z[tok]))
        out = self.up(out) if self.latent else out
        if self.shared is not None:
            out = out + self.shared(x)
        self.load = torch.bincount(sel.flatten(), minlength=self.E).float() / sel.numel()
        return out.view(shp)


def max_violation(load):
    """MaxVio = (max load - mean load) / mean load  (0 = perfectly balanced)."""
    return float((load.max() - load.mean()) / load.mean())


def quantile_offsets(scores, k):
    """Quantile balancing, exact one-shot version.  scores [m,n] (m tokens, n experts).
    Expert j gets the offset beta_j = the (m*k/n + 1)-th largest score in its column, i.e. the (1-k/n) quantile.
    An expert is switched on for a token when score - beta_j > 0 (a sign test, not a top-k sort), so
    every expert is on for ~ m*k/n tokens and the number of experts per token floats around k."""
    m, n = scores.shape
    r = int(round(m * k / n))
    return scores.sort(dim=0, descending=True).values[r]


def quantile_init(sigma, k, n):
    """beta_init = sigma * Phi^-1(1 - k/n) for logits assumed ~ N(0, sigma^2)."""
    return sigma * torch.distributions.Normal(0.0, 1.0).icdf(torch.tensor(1 - k / n)).item()
# <<end>>


# <<seg:attnres>>
class DepthAttn(nn.Module):
    """One attention-over-depth step (paper's block_attn_res).  A learned pseudo-query w (one vector per
    sublayer) scores the RMS-normalised earlier representations; softmax over the depth axis; weighted sum."""
    def __init__(self, d):
        super().__init__()
        self.w = nn.Parameter(torch.zeros(d))       # zero init => uniform average over sources at the start
        self.norm = RMSNorm(d)

    def forward(self, sources):
        V = torch.stack(sources)                      # [N, B, T, d]
        logits = torch.einsum("d,nbtd->nbt", self.w, self.norm(V))
        a = logits.softmax(0)
        self.last = a.detach()
        return torch.einsum("nbt,nbtd->btd", a, V)
# <<end>>


# <<seg:lm>>
class Block(nn.Module):
    def __init__(self, d, attn, ffn):
        super().__init__()
        self.n1, self.n2, self.attn, self.ffn = RMSNorm(d), RMSNorm(d), attn, ffn


class MiniLM(nn.Module):
    """Small decoder LM with pluggable attention / FFN and three residual styles:
        res="std"   h = h + f(norm h)                        (the usual residual stream)
        res="full"  every sublayer reads a softmax mix of ALL earlier sublayer outputs      (Full AttnRes)
        res="block" sublayer outputs are summed inside blocks; the mix runs over block sums (Block AttnRes)"""
    def __init__(self, vocab, d, make_attn, make_ffn, n_layers=4, res="std", n_blocks=None):
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        self.blocks = nn.ModuleList([Block(d, make_attn(i), make_ffn(i)) for i in range(n_layers)])
        self.norm = RMSNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)
        self.res, self.L = res, n_layers
        self.n_sub = 2 * n_layers
        if res != "std":
            self.n_blocks = n_blocks or self.n_sub
            self.sub_per_block = max(1, self.n_sub // self.n_blocks)
            self.mix = nn.ModuleList([DepthAttn(d) for _ in range(self.n_sub + 1)])   # +1: final output mix
        self.trace_on, self.tr = False, None       # set trace_on=True to record per-sublayer input / output norms in self.tr

    def _rec(self, h_in, f_out):
        if self.trace_on:
            self.tr["in"].append(h_in.detach().norm(dim=-1).mean().item())
            self.tr["out"].append(f_out.detach().norm(dim=-1).mean().item())

    def forward(self, idx, **kw):
        h0 = self.emb(idx)
        self.tr = {"in": [], "out": []}
        if self.res == "std":
            h = h0
            for b in self.blocks:
                f = b.attn(b.n1(h), **kw); self._rec(h, f); h = h + f
                f = b.ffn(b.n2(h)); self._rec(h, f); h = h + f
            return self.head(self.norm(h))
        outs = [h0]                      # Full: every sublayer output.  Block: finished block sums (+ embedding)
        partial, cnt, m = None, 0, 0
        for b in self.blocks:
            for kind in ("attn", "ffn"):
                srcs = outs if res_is_full(self) else outs + ([partial] if partial is not None else [])
                h = self.mix[m](srcs)
                m += 1
                f = b.attn(b.n1(h), **kw) if kind == "attn" else b.ffn(b.n2(h))
                self._rec(h, f)
                if self.res == "full":
                    outs.append(f)
                else:
                    partial = f if partial is None else partial + f
                    cnt += 1
                    if cnt == self.sub_per_block:
                        outs.append(partial)
                        partial, cnt = None, 0
        srcs = outs if res_is_full(self) else outs + ([partial] if partial is not None else [])
        return self.head(self.norm(self.mix[m](srcs)))


def res_is_full(model):
    return model.res == "full"


def load_corpus():
    return (Path(__file__).parent / "corpus.txt").read_text()


class CharData:
    def __init__(self, text=None, val_frac=0.1):
        text = load_corpus() if text is None else text
        chars = sorted(set(text))
        self.stoi = {c: i for i, c in enumerate(chars)}
        self.itos = chars
        ids = torch.tensor([self.stoi[c] for c in text], dtype=torch.long)
        n = int(len(ids) * (1 - val_frac))
        self.train, self.val = ids[:n], ids[n:]
        self.vocab = len(chars)

    def batch(self, split, B, T, gen):
        data = self.train if split == "train" else self.val
        ix = torch.randint(0, len(data) - T - 1, (B,), generator=gen)
        return torch.stack([data[i:i + T + 1] for i in ix])


def lm_loss(model, seq, **kw):
    logits = model(seq[:, :-1], **kw)
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), seq[:, 1:].reshape(-1))


@torch.no_grad()
def evaluate(model, data, T=64, n_batches=20, B=16, seed=123, **kw):
    g = torch.Generator().manual_seed(seed)
    model.eval()
    tot = sum(float(lm_loss(model, data.batch("val", B, T, g), **kw)) for _ in range(n_batches)) / n_batches
    model.train()
    return tot


def train_lm(model, data, steps=300, B=32, T=64, lr=3e-3, seed=0, hook=None, **kw):
    """Plain AdamW with linear warm-up and cosine decay. hook(step, model, loss) may add an extra loss."""
    g = torch.Generator().manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01, betas=(0.9, 0.95))
    losses = []
    for s in range(steps):
        cur = lr * min(1, (s + 1) / 20) * (0.5 * (1 + math.cos(math.pi * s / steps)) * 0.9 + 0.1)
        for gp in opt.param_groups:
            gp["lr"] = cur
        seq = data.batch("train", B, T, g)
        loss = lm_loss(model, seq, **kw)
        extra = hook(s, model, seq) if hook else 0.0
        opt.zero_grad()
        (loss + extra).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(loss.item())
    return losses
# <<end>>
