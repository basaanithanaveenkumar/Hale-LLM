"""Tiny single-process simulator of data-parallel training collectives (DDP / ZeRO / FSDP).

Every "rank" is a Python object in one process; collectives really move/sum tensors and count the
bytes each rank would send on a ring. This is a teaching tool: numerics are real, timing is not.
"""
import math, torch, torch.nn as nn
from torch.func import functional_call

F32 = 4  # bytes per fp32 element


# ----------------------------------------------------------------------------- collectives
class Comm:
    """Ring collectives over N simulated ranks; counts bytes *sent per rank*."""
    def __init__(self, N): self.N, self.sent, self.log = N, 0.0, []
    def _count(self, name, nbytes): self.sent += nbytes; self.log.append((name, nbytes))
    def all_reduce(self, xs):                       # every rank gets the sum
        s = sum(xs); self._count("all_reduce", 2 * (self.N - 1) / self.N * s.numel() * F32); return s
    def all_gather(self, shards):                   # every rank gets the concatenation
        full = torch.cat(shards); self._count("all_gather", (self.N - 1) / self.N * full.numel() * F32); return full
    def reduce_scatter(self, fulls):                # rank r gets slice r of the sum
        s = sum(fulls); self._count("reduce_scatter", (self.N - 1) / self.N * s.numel() * F32)
        return list(s.chunk(self.N))


def ring_all_reduce(vecs):
    """Explicit ring all-reduce. Returns (result per rank, bytes sent per rank, per-step log)."""
    N = len(vecs); chunks = [list(v.chunk(N)) for v in vecs]; sent = [0] * N; log = []
    for phase in ("reduce-scatter", "all-gather"):
        for step in range(N - 1):
            msgs = []
            for r in range(N):                      # rank r sends one chunk to rank r+1
                c = (r - step) % N if phase == "reduce-scatter" else (r + 1 - step) % N
                msgs.append((r, (r + 1) % N, c, chunks[r][c].clone()))
            for src, dst, c, payload in msgs:
                sent[src] += payload.numel() * F32
                chunks[dst][c] = chunks[dst][c] + payload if phase == "reduce-scatter" else payload
            log.append((phase, step, [(s, d, c) for s, d, c, _ in msgs]))
    return [torch.cat(c) for c in chunks], sent, log


# ----------------------------------------------------------------------------- model / units
D_IN, D_H, D_OUT = 16, 64, 4

def make_layers(seed=0):
    torch.manual_seed(seed)
    return [nn.Linear(D_IN, D_H), nn.Tanh(), nn.Linear(D_H, D_H), nn.Tanh(),
            nn.Linear(D_H, D_H), nn.Tanh(), nn.Linear(D_H, D_OUT)]

def make_model(seed=0): return nn.Sequential(*make_layers(seed))

def make_units(seed=0, per="layer"):
    """Group consecutive layers into FSDP 'units'. per='layer' -> 4 units, 'pair' -> 2, 'model' -> 1."""
    L = make_layers(seed)
    groups = {"layer": [[0, 1], [2, 3], [4, 5], [6]], "pair": [[0, 1, 2, 3], [4, 5, 6]], "model": [list(range(7))]}[per]
    return [nn.Sequential(*[L[i] for i in g]) for g in groups]

def make_data(N, per_rank=16, seed=1):
    torch.manual_seed(seed); X = torch.randn(N * per_rank, D_IN); Y = torch.randn(N * per_rank, D_OUT)
    return list(X.chunk(N)), list(Y.chunk(N)), X, Y

def nparams(m): return sum(p.numel() for p in m.parameters())
def flat_params(units): return torch.cat([p.detach().flatten() for u in units for p in u.parameters()])

def _flat(unit): return torch.cat([p.detach().flatten() for p in unit.parameters()])
def _unflat(unit, flat):
    d, o = {}, 0
    for n, p in unit.named_parameters(): d[n] = flat[o:o + p.numel()].view_as(p); o += p.numel()
    return d
def _pad(t, N): k = (-t.numel()) % N; return torch.cat([t, t.new_zeros(k)]) if k else t


# ----------------------------------------------------------------------------- memory tracker
class Tracker:
    """Records rank-0 live bytes by category after every event."""
    CATS = ("params", "grads", "optimizer", "gathered params", "gathered grads")
    def __init__(self): self.events, self.transient = [], {"gathered params": 0, "gathered grads": 0}
    def snap(self, name, persistent):
        row = dict(persistent); row.update(self.transient)
        self.events.append((name, {c: row.get(c, 0) for c in self.CATS}))
    @property
    def peak(self): return max(sum(r.values()) for _, r in self.events)


def _opt_bytes(opt):
    return sum(v.numel() * F32 for s in opt.state.values() for v in s.values() if torch.is_tensor(v) and v.dim() > 0)

def _grad_bytes(ps): return sum(p.grad.numel() * F32 for p in ps if p.grad is not None)


# ----------------------------------------------------------------------------- baseline
def baseline_train(steps, lr, N=4, per_rank=16):
    """Single process, one big batch of N*per_rank samples (what DDP/FSDP must reproduce)."""
    m = make_model(); _, _, X, Y = make_data(N, per_rank); opt = torch.optim.Adam(m.parameters(), lr=lr); losses = []
    for _ in range(steps):
        opt.zero_grad(); loss = ((m(X) - Y) ** 2).mean(); loss.backward(); opt.step(); losses.append(loss.item())
    return torch.cat([p.detach().flatten() for p in m.parameters()]), losses


# ----------------------------------------------------------------------------- DDP
class DDPSim:
    def __init__(self, N, lr, per="layer"):
        self.N, self.comm = N, Comm(N)
        self.units = [make_units(per=per) for _ in range(N)]                    # N identical replicas
        self.models = [nn.Sequential(*[l for u in us for l in u]) for us in self.units]
        self.opts = [torch.optim.Adam(m.parameters(), lr=lr) for m in self.models]
        self.tr = Tracker()
    def _persist(self, r=0):
        ps = list(self.models[r].parameters())
        return {"params": nparams(self.models[r]) * F32, "grads": _grad_bytes(ps), "optimizer": _opt_bytes(self.opts[r])}
    def step(self, xs, ys):
        N, tr = self.N, self.tr; tr.snap("start of step", self._persist())
        losses = []
        for r in range(N):                                                       # 1) local forward+backward
            self.opts[r].zero_grad(); loss = ((self.models[r](xs[r]) - ys[r]) ** 2).mean(); loss.backward(); losses.append(loss.item())
        tr.snap("local backward done (full grads)", self._persist())
        ps = [list(m.parameters()) for m in self.models]
        for j in range(len(ps[0])):                                              # 2) all-reduce grads, average
            g = self.comm.all_reduce([ps[r][j].grad.flatten() for r in range(N)]) / N
            for r in range(N): ps[r][j].grad = g.view_as(ps[r][j]).clone()
        tr.snap("all-reduce grads", self._persist())
        for r in range(N): self.opts[r].step()                                   # 3) identical optimizer step
        tr.snap("optimizer step", self._persist())
        return sum(losses) / N
    def flat(self, r=0): return torch.cat([p.detach().flatten() for p in self.models[r].parameters()])


# ----------------------------------------------------------------------------- ZeRO / FSDP
class ShardSim:
    """ZeRO-style sharded data parallelism on N simulated ranks.
    stage 1: shard optimizer state; stage 2: + grads; stage 3 (== FSDP FULL_SHARD): + params."""
    def __init__(self, N, lr, stage=3, per="layer"):
        self.N, self.stage, self.comm, self.tr = N, stage, Comm(N), Tracker()
        self.units = make_units(per=per)                                         # structure + init (template)
        self.n = [nparams(u) for u in self.units]
        self.shards = []                                                         # shards[r][u]: owned slice of unit u
        for r in range(N):
            row = []
            for u, unit in enumerate(self.units):
                f = _pad(_flat(unit), N); row.append(nn.Parameter(f.chunk(N)[r].clone()))
            self.shards.append(row)
        # stages 1/2 keep a full replica of params on every rank (no param sharding)
        self.full = [[_pad(_flat(u), N).clone() for u in self.units] for _ in range(N)] if stage < 3 else None
        self.opts = [torch.optim.Adam(self.shards[r], lr=lr) for r in range(N)]
    # --- helpers
    def _persist(self):
        sh = self.shards[0]; N = self.N
        params = sum(self.n[u] for u in range(len(sh))) * F32 if self.stage < 3 else sum(p.numel() for p in sh) * F32
        grads = _grad_bytes(sh) if self.stage >= 2 else self._held_full_grads
        return {"params": params, "grads": grads, "optimizer": _opt_bytes(self.opts[0])}
    _held_full_grads = 0
    def _gather(self, u):
        full = self.comm.all_gather([self.shards[r][u].data for r in range(self.N)])
        return full[:self.n[u]]
    def step(self, xs, ys):
        N, tr, U, st = self.N, self.tr, len(self.units), self.stage
        self._held_full_grads = 0; held = {}; tr.snap("start of step", self._persist())
        acts, cur = [[None] * U for _ in range(N)], list(xs)
        for u, unit in enumerate(self.units):                                    # ---- forward
            if st == 3:
                full = self._gather(u); tr.transient["gathered params"] = full.numel() * F32; tr.snap(f"all-gather unit {u}", self._persist())
            else: full = self.full_unit(u)
            for r in range(N):
                acts[r][u] = cur[r]
                with torch.no_grad(): cur[r] = functional_call(unit, _unflat(unit, full), (cur[r],))
            tr.snap(f"forward unit {u}", self._persist())
            if st == 3: tr.transient["gathered params"] = 0; tr.snap(f"free unit {u}", self._persist())
        gs, losses = [], []
        for r in range(N):
            o = cur[r].detach().requires_grad_(); loss = ((o - ys[r]) ** 2).mean(); loss.backward(); gs.append(o.grad); losses.append(loss.item())
        for u in reversed(range(U)):                                             # ---- backward
            unit = self.units[u]
            if st == 3:
                full = self._gather(u); tr.transient["gathered params"] = full.numel() * F32; tr.snap(f"re-gather unit {u}", self._persist())
            else: full = self.full_unit(u)
            gl = []
            for r in range(N):
                p = full.clone().requires_grad_(); xin = acts[r][u].detach().requires_grad_()
                functional_call(unit, _unflat(unit, p), (xin,)).backward(gs[r]); gl.append(_pad(p.grad, N)); gs[r] = xin.grad
            tr.transient["gathered grads"] = gl[0].numel() * F32; tr.snap(f"backward unit {u} (full grad)", self._persist())
            if st == 1:                                                          # stage 1: keep full grads until backward ends
                held[u] = gl; self._held_full_grads += gl[0].numel() * F32
            else:                                                                # stage 2/3: reduce-scatter
                parts = self.comm.reduce_scatter(gl)
                for r in range(N): self.shards[r][u].grad = parts[r] / N
            tr.transient["gathered grads"] = 0; tr.transient["gathered params"] = 0
            tr.snap(f"reduce-scatter unit {u}" if st > 1 else f"keep full grad of unit {u}", self._persist())
        if st == 1:                                                              # stage 1: reduce-scatter once, at the end
            for u in range(U):
                parts = self.comm.reduce_scatter(held[u])
                for r in range(N): self.shards[r][u].grad = parts[r] / N
            tr.snap("reduce-scatter all grads", self._persist()); self._held_full_grads = 0
        for r in range(N): self.opts[r].step()
        if st < 3:                                                               # stages 1/2: all-gather updated params
            for u in range(U):
                new = self.comm.all_gather([self.shards[r][u].data for r in range(N)])
                for r in range(N): self.full[r][u] = new.clone()
        tr.snap("optimizer step on shard", self._persist())
        for r in range(N):
            for p in self.shards[r]: p.grad = None
        return sum(losses) / N
    def full_unit(self, u): return self.full[0][u][:self.n[u]]
    def flat(self):
        return torch.cat([torch.cat([self.shards[r][u].data for r in range(self.N)])[:self.n[u]] for u in range(len(self.units))])


# ----------------------------------------------------------------------------- bytes formulas
def state_bytes_per_gpu(psi, N, stage, k_opt=12, p_bytes=2, g_bytes=2):
    """ZeRO paper accounting (Rajbhandari et al., 2020). Mixed precision + Adam: k_opt=12 (fp32 master, m, v)."""
    p = p_bytes * psi / (N if stage >= 3 else 1)
    g = g_bytes * psi / (N if stage >= 2 else 1)
    o = k_opt * psi / (N if stage >= 1 else 1)
    return p + g + o
