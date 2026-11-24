"""fa_kit: shared helpers for the FlashAttention-variants notebooks.

Nothing here implements an attention algorithm. Every algorithm (standard,
FlashAttention-1/2/3, Flash-Decoding, masks, Ring) is written out in the
notebooks themselves; this module only holds

* ``ref_attention``  - a float64 reference used to *check* the from-scratch code
* ``IOCounter``      - counts simulated HBM (slow memory) element reads/writes
* ``FlopCounter``    - counts matmul / exp FLOPs of the simulated kernels
* ``check``/``report`` - tiny assertion registry (prints a pass/fail table)
* drawing primitives (``box``, ``arrow``, ``tile_grid``) with one palette

Design note: we simulate the GPU memory hierarchy by *counting* the elements a
kernel must move between slow memory (HBM) and fast memory (SRAM/registers).
This is exact for the algorithm as written but it is not a wall-clock
measurement; the notebooks say so wherever it matters.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

# --------------------------------------------------------------------------
# device: CUDA if available, else CPU. Override with env FA_DEVICE=cpu|cuda|cuda:1.
# set_default_device makes every torch.randn/zeros/arange/tensor in the notebooks
# land on it, so the notebook code stays device-agnostic. TF32 is switched off so
# float32 matmuls keep full precision and the exactness checks mean the same thing
# on a GPU as on a CPU (float64 is used for the tight 1e-11 checks).
# --------------------------------------------------------------------------
DEVICE = os.environ.get("FA_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device(DEVICE)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def sync():
    """Wait for queued GPU work (needed before wall-clock timing); no-op on CPU."""
    if torch.device(DEVICE).type == "cuda":
        torch.cuda.synchronize()


# --------------------------------------------------------------------------
# palette (one colour per tensor, reused in every figure in every notebook)
# --------------------------------------------------------------------------
COL = dict(
    q="#4C78A8", k="#F58518", v="#54A24B", s="#B279A2", p="#9D7660",
    o="#E45756", hbm="#BAB0AC", sram="#72B7B2", ink="#222222", soft="#F2F2F2",
    good="#2CA02C", bad="#D62728", mid="#FFBF00",
)
plt.rcParams.update({
    "figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 10, "axes.titleweight": "bold", "axes.titlesize": 11,
})


# --------------------------------------------------------------------------
# reference
# --------------------------------------------------------------------------
def ref_attention(q, k, v, causal=False, bias=None, mask=None, scale=None):
    """float64 reference attention. q:(...,Nq,d) k,v:(...,Nk,d).

    Returns (out, lse) where lse is the row-wise log-sum-exp of the scaled
    scores - the single statistic FlashAttention stores for the backward pass.
    ``causal`` aligns the *last* query with the last key (bottom-right), the
    convention used for decoding with a KV cache.
    """
    q, k, v = q.double(), k.double(), v.double()
    scale = scale if scale is not None else 1.0 / math.sqrt(q.shape[-1])
    s = (q @ k.transpose(-1, -2)) * scale
    if bias is not None:
        s = s + bias.double()
    nq, nk = s.shape[-2:]
    if causal:
        m = torch.ones(nq, nk, dtype=torch.bool).tril(diagonal=nk - nq)
        s = s.masked_fill(~m, float("-inf"))
    if mask is not None:
        s = s.masked_fill(~mask, float("-inf"))
    lse = torch.logsumexp(s, dim=-1)
    p = torch.exp(s - lse[..., None])
    return p @ v, lse


def max_err(a, b):
    return (a.double() - b.double()).abs().max().item()


# --------------------------------------------------------------------------
# counters
# --------------------------------------------------------------------------
@dataclass
class IOCounter:
    """Counts elements moved between HBM and on-chip memory."""
    reads: dict = field(default_factory=dict)
    writes: dict = field(default_factory=dict)

    def read(self, name, t_or_n):
        n = t_or_n.numel() if hasattr(t_or_n, "numel") else int(t_or_n)
        self.reads[name] = self.reads.get(name, 0) + n

    def write(self, name, t_or_n):
        n = t_or_n.numel() if hasattr(t_or_n, "numel") else int(t_or_n)
        self.writes[name] = self.writes.get(name, 0) + n

    @property
    def total(self):
        return sum(self.reads.values()) + sum(self.writes.values())

    def by_tensor(self):
        names = sorted(set(self.reads) | set(self.writes))
        return {n: self.reads.get(n, 0) + self.writes.get(n, 0) for n in names}


@dataclass
class FlopCounter:
    matmul: int = 0     # 2*m*n*k per GEMM
    nonmatmul: int = 0  # exp / scale / max / rescale element ops

    def mm(self, m, n, k):
        self.matmul += 2 * m * n * k

    def elt(self, n):
        self.nonmatmul += n


# --------------------------------------------------------------------------
# checks
# --------------------------------------------------------------------------
_CHECKS: list = []


def check(name, cond, detail=""):
    _CHECKS.append((name, bool(cond), detail))
    assert cond, f"CHECK FAILED: {name} {detail}"
    return cond


def report():
    ok = sum(c for _, c, _ in _CHECKS)
    for n, c, d in _CHECKS:
        print(("PASS " if c else "FAIL ") + n + (f"   [{d}]" if d else ""))
    print(f"\n{ok}/{len(_CHECKS)} checks passed")


# --------------------------------------------------------------------------
# drawing primitives
# --------------------------------------------------------------------------
def box(ax, x, y, w, h, text="", color="#EEEEEE", ec="#444444", fs=9, tc=None, lw=1.2, alpha=1.0, round_=0.02):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={round_}",
                                fc=color, ec=ec, lw=lw, alpha=alpha))
    if text:
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
                color=tc or COL["ink"])


def arrow(ax, p0, p1, color="#444444", lw=1.4, style="-|>", rad=0.0, ls="-"):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=11, lw=lw, color=color,
                                 connectionstyle=f"arc3,rad={rad}", linestyle=ls))


def canvas(w=9, h=4, xlim=(0, 10), ylim=(0, 5), title=None):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.set_aspect("auto"); ax.axis("off")
    if title:
        ax.set_title(title, loc="left")
    return fig, ax


def tile_grid(ax, nr, nc, order=None, mask=None, cmap="Blues", label_order=True, title=None,
              rlabel="Q block i", clabel="K/V block j", skip_color="#E8E8E8", fs=8):
    """Draw an nr x nc grid of score tiles S[i,j].

    order : dict {(i,j): step} colours a tile by visit order and prints the step.
    mask  : bool array (nr,nc); False tiles are drawn as skipped (hatched grey).
    """
    n = max(order.values()) if order else 1
    cm = plt.get_cmap(cmap)
    for i in range(nr):
        for j in range(nc):
            live = True if mask is None else bool(mask[i][j])
            if not live:
                ax.add_patch(Rectangle((j, nr - 1 - i), 1, 1, fc=skip_color, ec="white", lw=1.5, hatch="///"))
                continue
            c = cm(0.25 + 0.7 * (order[(i, j)] - 1) / max(n - 1, 1)) if order and (i, j) in order else cm(0.35)
            ax.add_patch(Rectangle((j, nr - 1 - i), 1, 1, fc=c, ec="white", lw=1.5))
            if order and label_order and (i, j) in order:
                ax.text(j + .5, nr - 1 - i + .5, str(order[(i, j)]), ha="center", va="center", fontsize=fs,
                        color="white" if order[(i, j)] > n * .35 else COL["ink"])
    ax.set_xlim(0, nc); ax.set_ylim(0, nr); ax.set_aspect("equal")
    ax.set_xticks(np.arange(nc) + .5); ax.set_xticklabels([f"{j}" for j in range(nc)], fontsize=7)
    ax.set_yticks(np.arange(nr) + .5); ax.set_yticklabels([f"{i}" for i in reversed(range(nr))], fontsize=7)
    ax.set_xlabel(clabel, fontsize=8); ax.set_ylabel(rlabel, fontsize=8)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    if title:
        ax.set_title(title, fontsize=9, loc="left")


def caption(text):
    """Print a figure caption that tells the reader what to notice."""
    from IPython.display import Markdown, display
    display(Markdown(f"<sub>**Figure.** {text}</sub>"))
