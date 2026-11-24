"""predkit: pure-PyTorch building blocks for the prediction-objectives notebook.

Contents
* ``star_graph_batch``   - the star-graph planning task of Bachmann & Nagarajan (2024)
* ``TinyGPT``            - a small pre-LN decoder-only transformer returning final hidden states
* four objectives as separate loss functions, each written to be read:
    ``loss_ntp``     next-token prediction
    ``loss_mtp``     independent multi-token heads (Gloeckle et al., 2024)
    ``loss_jtp``     joint multi-token prediction with a Fetch bottleneck (Ahn et al., 2025)
    ``loss_nextlat`` next-latent prediction (Teoh et al., 2025)
* ``solve_rate``      - greedy-decode the path and score it exactly
"""
from __future__ import annotations

import copy
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

def _f(x):
    return float(x.detach()) if torch.is_tensor(x) else float(x)


DEVICE = os.environ.get("PK_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------------------------------- data
class StarGraph:
    """G(d, l): d paths of l nodes sharing one centre node. Node labels drawn from {0..N-1}.

    Sequence = [edge list (2E tokens) | SEP | start goal | SEP | path (l tokens)].
    Only the path tokens carry loss. The goal sits on exactly one of the d branches;
    picking the right branch at the centre needs the goal, which the *next-token*
    objective gives no incentive to look back at (the "Clever Hans" shortcut).
    """

    def __init__(self, d=2, l=5, n_nodes=50):
        self.d, self.l, self.N = d, l, n_nodes
        self.E = d * (l - 1)
        self.SEP = n_nodes
        self.vocab = n_nodes + 1
        self.prefix = 2 * self.E + 1 + 2 + 1          # tokens before the path
        self.T = self.prefix + l

    def batch(self, B, device=None):
        d, l, N, E = self.d, self.l, self.N, self.E
        device = device or DEVICE
        M = d * (l - 1) + 1
        perm = torch.rand(B, N, device=device).argsort(1)[:, :M]
        centre = perm[:, :1]
        branches = perm[:, 1:].reshape(B, d, l - 1)
        paths = torch.cat([centre[:, None, :].expand(B, d, 1), branches], 2)          # (B,d,l)
        edges = torch.stack([paths[:, :, :-1], paths[:, :, 1:]], -1).reshape(B, E, 2)   # (B,E,2)
        order = torch.rand(B, E, device=device).argsort(1)
        edges = torch.gather(edges, 1, order[:, :, None].expand(B, E, 2)).reshape(B, 2 * E)
        pick = torch.randint(0, d, (B,), device=device)
        path = paths[torch.arange(B, device=device), pick]                             # (B,l)
        sep = torch.full((B, 1), self.SEP, device=device)
        seq = torch.cat([edges, sep, path[:, :1], path[:, -1:], sep, path], 1)
        return seq


# --------------------------------------------------------------------------- model
class Block(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim))

    def forward(self, x):
        T = x.shape[1]
        m = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), 1)
        a = self.ln1(x)
        x = x + self.attn(a, a, a, attn_mask=m, need_weights=False)[0]
        return x + self.mlp(self.ln2(x))


class TinyGPT(nn.Module):
    """Decoder-only transformer. ``forward`` returns final hidden states h (B,T,dim); ``head`` maps h to logits."""

    def __init__(self, vocab, max_len, dim=128, layers=4, heads=4):
        super().__init__()
        self.dim = dim
        self.tok, self.pos = nn.Embedding(vocab, dim), nn.Embedding(max_len, dim)
        self.blocks = nn.ModuleList(Block(dim, heads) for _ in range(layers))
        self.ln_f = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab, bias=False)
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)
            if getattr(m, "bias", None) is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x):
        h = self.tok(x) + self.pos(torch.arange(x.shape[1], device=x.device))
        for b in self.blocks:
            h = b(h)
        return self.ln_f(h)


def path_mask(sg: StarGraph, T_in):
    """Boolean (T_in,) mask over *target* positions: True where the target is a path token."""
    tgt_idx = torch.arange(1, T_in + 1, device=DEVICE)
    return tgt_idx >= sg.prefix


def shifted_ce(logits, targets, mask):
    """Mean cross-entropy over positions where ``mask`` is True. logits (B,T,V), targets (B,T), mask (B,T)."""
    ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), reduction="none").view_as(targets)
    return (ce * mask).sum() / mask.sum().clamp_min(1)


# --------------------------------------------------------------------------- objectives
def loss_ntp(model, seq, sg, **_):
    """Next-token prediction: position p predicts x_{p+1}."""
    x, y = seq[:, :-1], seq[:, 1:]
    h = model(x)
    m = path_mask(sg, x.shape[1])[None].expand_as(y).float()
    ln = shifted_ce(model.head(h), y, m)
    return ln, {"ntp": ln.item()}


class MTPHeads(nn.Module):
    """Gloeckle et al.: D extra heads, each = one transformer block on the shared trunk output, shared unembedding.
    Head i (1..D) predicts x_{p+1+i} from h_p *independently* of the other heads and of x_{p+1..}."""

    def __init__(self, dim, heads, D):
        super().__init__()
        self.blocks = nn.ModuleList(Block(dim, heads) for _ in range(D))


def loss_mtp(model, seq, sg, extra: MTPHeads, D, lam=1.0, **_):
    x, y = seq[:, :-1], seq[:, 1:]
    h = model(x)
    B, T = y.shape
    pm = path_mask(sg, T)[None].expand(B, T).float()
    ln = shifted_ce(model.head(h), y, pm)
    aux = 0.0
    for i in range(1, D + 1):
        hi = extra.blocks[i - 1](h)                       # head i: its own block, same trunk input
        tgt = y[:, i:]                                    # x_{p+1+i}
        msk = pm[:, i:]
        aux = aux + shifted_ce(model.head(hi)[:, :T - i], tgt, msk) / D
    return ln + lam * aux, {"ntp": ln.item(), "mtp": _f(aux)}


class Fetch(nn.Module):
    """Ahn et al.: u_j = gamma*h + Emb(x_{p+j}) for j=0..D; Fetch_j(h) = h + SelfAttn(u_0..u_j) (causal over j)."""

    def __init__(self, dim, heads, gamma=1.0):
        super().__init__()
        self.gamma = gamma
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.ln = nn.LayerNorm(dim)

    def forward(self, h, emb_win):
        # h: (N,dim)   emb_win: (N, D+1, dim)  -> out (N, D+1, dim)
        u = self.ln(self.gamma * h[:, None, :] + emb_win)
        J = u.shape[1]
        m = torch.triu(torch.ones(J, J, dtype=torch.bool, device=u.device), 1)
        return h[:, None, :] + self.attn(u, u, u, attn_mask=m, need_weights=False)[0]


def loss_jtp(model, seq, sg, fetch: Fetch, D, lam=1.0, **_):
    x, y = seq[:, :-1], seq[:, 1:]
    h = model(x)
    B, T = y.shape
    pm = path_mask(sg, T)[None].expand(B, T).float()
    ln = shifted_ce(model.head(h), y, pm)
    # windows of teacher-forced input embeddings: win[b,p,j] = Emb(x[b,p+j]), j=0..D   (right-padded; masked out below)
    pad = F.pad(x, (0, D), value=0)
    win = torch.stack([pad[:, j:j + T] for j in range(D + 1)], 2)             # (B,T,D+1)
    ew = model.tok(win).reshape(B * T, D + 1, -1)
    out = fetch(h.reshape(B * T, -1), ew).reshape(B, T, D + 1, -1)           # out[:,:,j] predicts x_{p+1+j}
    aux = 0.0
    for j in range(1, D + 1):
        valid = pm[:, j:]
        aux = aux + shifted_ce(model.head(out[:, :T - j, j]), y[:, j:], valid) / D
    return ln + lam * aux, {"ntp": ln.item(), "jtp": _f(aux)}


class Dynamics(nn.Module):
    """Teoh et al.: p_psi(h_t, x_{t+1}) -> h_hat_{t+1}. A plain MLP with a residual connection."""

    def __init__(self, dim, vocab_emb: nn.Embedding):
        super().__init__()
        self.emb = vocab_emb
        self.net = nn.Sequential(nn.Linear(2 * dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim))

    def forward(self, h, tok):
        return h + self.net(torch.cat([h, self.emb(tok)], -1))


def loss_nextlat(model, seq, sg, dyn: Dynamics, horizon=1, lam_h=1.0, lam_kl=1.0, **_):
    x, y = seq[:, :-1], seq[:, 1:]
    h = model(x)
    B, T = y.shape
    pm = path_mask(sg, T)[None].expand(B, T).float()
    logits = model.head(h)
    ln = shifted_ce(logits, y, pm)
    frozen = copy.deepcopy(model.head).requires_grad_(False)          # frozen output head for the KL term
    cur = h
    loss_h = loss_kl = 0.0
    for i in range(1, horizon + 1):
        cur = cur[:, :-1]                                              # h_t       (t = 0 .. T-1-i)
        nxt_tok = x[:, i:]                                             # x_{t+i} fed as the "action"
        tgt_h = h[:, i:].detach()                                      # sg[h_{t+i}]
        pred = dyn(cur, nxt_tok)                                       # h_hat_{t+i}
        msk = pm[:, i:i + pred.shape[1]]                               # h_{t+i} predicts x_{t+i+1}: supervise where that target is a path token
        sl1 = F.smooth_l1_loss(pred, tgt_h, reduction="none").mean(-1)
        loss_h = loss_h + (sl1 * msk).sum() / msk.sum().clamp_min(1)
        p_true = F.softmax(logits[:, i:].detach(), -1)
        logq = F.log_softmax(frozen(pred), -1)
        kl = (p_true * (p_true.clamp_min(1e-9).log() - logq)).sum(-1)
        loss_kl = loss_kl + (kl * msk).sum() / msk.sum().clamp_min(1)
        cur = pred                                                     # recursive roll-out
    loss_h, loss_kl = loss_h / horizon, loss_kl / horizon
    return ln + lam_h * loss_h + lam_kl * loss_kl, {"ntp": ln.item(), "next_h": _f(loss_h), "kl": _f(loss_kl)}


# --------------------------------------------------------------------------- evaluation
@torch.no_grad()
def solve_rate(model, sg: StarGraph, n=512):
    """Greedy-decode the l path tokens from the prompt; the answer is correct only if the whole path matches."""
    model.eval()
    seq = sg.batch(n)
    cur = seq[:, :sg.prefix].clone()
    for _ in range(sg.l):
        nxt = model.head(model(cur))[:, -1].argmax(-1, keepdim=True)
        cur = torch.cat([cur, nxt], 1)
    model.train()
    return (cur[:, sg.prefix:] == seq[:, sg.prefix:]).all(1).float().mean().item()


# --------------------------------------------------------------------------- experiment driver
def build_method(name, sg: StarGraph, dim=128, layers=3, heads=4, D=None, seed=0):
    """Returns (model, extra_modules dict, loss_fn, loss_kwargs). D defaults to l-2 as in Ahn et al. (2025)."""
    torch.manual_seed(seed)
    D = D if D is not None else sg.l - 2
    model = TinyGPT(sg.vocab, sg.T, dim=dim, layers=layers, heads=heads).to(DEVICE)
    if name == "NTP":
        return model, {}, loss_ntp, {}
    if name == "MTP":
        ex = MTPHeads(dim, heads, D).to(DEVICE)
        return model, {"extra": ex}, loss_mtp, {"extra": ex, "D": D}
    if name == "JTP":
        fe = Fetch(dim, heads).to(DEVICE)
        return model, {"fetch": fe}, loss_jtp, {"fetch": fe, "D": D}
    if name == "NextLat":
        dy = Dynamics(dim, model.tok).to(DEVICE)
        return model, {"dyn": dy}, loss_nextlat, {"dyn": dy, "horizon": D}
    raise ValueError(name)


def n_params(model, extra):
    seen, tot = set(), 0
    for m in [model, *extra.values()]:
        for p in m.parameters():
            if id(p) not in seen:
                seen.add(id(p)); tot += p.numel()
    return tot


def train_method(name, steps=4000, batch=128, lr=3e-4, eval_every=250, seed=0, d=2, l=5, n_nodes=50, dim=128, layers=3, log=print):
    import time
    sg = StarGraph(d, l, n_nodes)
    model, extra, fn, kw = build_method(name, sg, dim, layers, seed=seed)
    params = list({id(p): p for m in [model, *extra.values()] for p in m.parameters()}.values())
    opt = torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.95), weight_decay=0.01)
    hist = {"step": [], "solve": [], "loss": []}
    t0 = time.time()
    for s in range(1, steps + 1):
        loss, info = fn(model, sg.batch(batch), sg, **kw)
        opt.zero_grad(); loss.backward(); opt.step()
        if s % eval_every == 0:
            sr = solve_rate(model, sg)
            hist["step"].append(s); hist["solve"].append(sr); hist["loss"].append(info)
            log(f"{name:8s} step {s:5d}  solve {sr:.3f}  {({k: round(v, 3) for k, v in info.items()})}  {time.time() - t0:.0f}s")
    sec = (time.time() - t0) / steps
    return dict(hist=hist, sec_per_step=sec, n_params=n_params(model, extra), model=model.state_dict(),
                extra={k: v.state_dict() for k, v in extra.items()})
