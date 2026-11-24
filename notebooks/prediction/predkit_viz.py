"""predkit_viz: figures and GIFs for the prediction-objectives notebook (matplotlib + Pillow only)."""
from __future__ import annotations

import io

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from PIL import Image

C = dict(inp="#D9D9D9", hid="#FFFFFF", tgt="#FFD58A", tf="#F28B82", ink="#222222", good="#2CA02C", bad="#D62728",
         ntp="#7F7F7F", mtp="#4C78A8", jtp="#54A24B", nl="#E45756", hi="#1F77B4")


def _box(ax, x, y, w, h, text, fc, ec="#444", fs=8, lw=1.2, alpha=1.0):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0,rounding_size=0.08", fc=fc, ec=ec, lw=lw, alpha=alpha))
    ax.text(x, y, text, ha="center", va="center", fontsize=fs)


def _arrow(ax, p0, p1, color="#555", ls="-", lw=1.4, rad=0.0, alpha=1.0):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=10, lw=lw, color=color, ls=ls, alpha=alpha,
                                 connectionstyle=f"arc3,rad={rad}", shrinkA=1, shrinkB=1))


def _frame_to_pil(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=90)
    buf.seek(0)
    return Image.open(buf).convert("P", palette=Image.ADAPTIVE)


def save_gif(frames, path, ms=900):
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=ms, loop=0, disposal=2)
    return path


# ----------------------------------------------------------------------------- GIF 1: what each objective supervises
def supervision_gif(path, T=8, D=2, positions=range(1, 6)):
    """Four panels (NTP / MTP / JTP / NextLat). Each frame highlights the hidden state h_p and draws exactly
    which targets it is trained on, and through which route (dashed red = teacher-forced token)."""
    names = ["Next-token", "Multi-token (independent heads)", "Joint multi-token (Fetch bottleneck)", "Next-latent"]
    cols = [C["ntp"], C["mtp"], C["jtp"], C["nl"]]
    frames = []
    for p in positions:
        fig, axs = plt.subplots(1, 4, figsize=(17, 4.6))
        for k, ax in enumerate(axs):
            ax.set_xlim(0.2, T + 1.8); ax.set_ylim(-0.3, 4.9); ax.axis("off")
            ax.set_title(names[k], fontsize=10, color=cols[k], weight="bold")
            xs = {i: i for i in range(1, T + 1)}
            for i in range(1, T + 1):
                inp_hi = i <= p
                _box(ax, xs[i], 0.3, 0.7, 0.5, f"x{i}", C["inp"] if inp_hi else "#F4F4F4", alpha=1 if inp_hi else .5)
                _box(ax, xs[i], 1.7, 0.7, 0.5, f"h{i}", "#CFE3F7" if i == p else C["hid"], ec=C["hi"] if i == p else "#999", lw=2.2 if i == p else 1, alpha=1 if i <= p else .35)
                _box(ax, xs[i] + 0.0, 4.3, 0.7, 0.5, f"x{i}", "#F4F4F4", alpha=.35)
                if i <= p:
                    _arrow(ax, (xs[i], 0.55), (xs[i], 1.45), alpha=.35 if i != p else .9)
            # history -> h_p
            tgt_range = {0: [p + 1], 1: [p + 1 + j for j in range(D + 1)], 2: [p + 1 + j for j in range(D + 1)], 3: [p + 1 + j for j in range(D + 1)]}[k]
            for t in tgt_range:
                if t > T:
                    continue
                _box(ax, t, 4.3, 0.7, 0.5, f"x̂{t}", C["tgt"], ec="#B8860B", lw=1.6)
            if k == 0:
                _arrow(ax, (p, 1.95), (p + 1, 4.05), color=cols[k], lw=2)
            if k == 1:
                for t in tgt_range:
                    if t <= T: _arrow(ax, (p, 1.95), (t, 4.05), color=cols[k], lw=2, rad=0.0)
                ax.text(T / 2 + 1, 2.9, "all heads read the same h", fontsize=8, ha="center", color=cols[k])
            if k == 2:
                _arrow(ax, (p, 1.95), (p + 1, 4.05), color=cols[k], lw=2)
                _box(ax, p + 1.4, 2.95, 1.7, 0.55, "Fetch(h, x)", "#DDF0DD", ec=cols[k], fs=8)
                _arrow(ax, (p + .1, 1.95), (p + .8, 2.7), color=cols[k], lw=2)
                for j in range(1, D + 1):
                    t = p + 1 + j
                    if t <= T: _arrow(ax, (p + 1.4 + (j - 1.5) * .3, 3.25), (t, 4.05), color=cols[k], lw=1.8)
                for j in range(1, D + 1):
                    s = p + j
                    if s <= T:
                        _box(ax, s, 0.3, 0.7, 0.5, f"x{s}", C["tf"], ec=C["bad"], lw=1.8)
                        _arrow(ax, (s, .58), (p + 1.4 + (j - 1.5) * .3 - .1, 2.65), color=C["bad"], ls="--", lw=1.2, rad=-.15)
            if k == 3:
                _arrow(ax, (p, 1.95), (p + 1, 4.05), color=cols[k], lw=2)
                prev = p
                for j in range(1, D + 1):
                    hx = p + j
                    if hx > T: break
                    _box(ax, hx + .5 if j == 1 else hx + .5, 2.95 + 0.0, 0.8, 0.5, f"ĥ{p + j}", "#FADBD8", ec=cols[k], fs=8)
                    _arrow(ax, (prev + (.0 if j == 1 else .5 + .4), 1.95 if j == 1 else 2.95), (hx + .5 - .4, 2.95), color=cols[k], lw=1.8)
                    s = p + j
                    _box(ax, s, 0.3, 0.7, 0.5, f"x{s}", C["tf"], ec=C["bad"], lw=1.8)
                    _arrow(ax, (s, .58), (hx + .5, 2.68), color=C["bad"], ls="--", lw=1.2, rad=-.1)
                    t = p + 1 + j
                    if t <= T: _arrow(ax, (hx + .5, 3.2), (t, 4.05), color=cols[k], lw=1.8)
                    prev = hx
            ax.text(0.3, 4.85, "targets", fontsize=7, color="#888"); ax.text(0.3, -0.25, "inputs (red = teacher-forced)", fontsize=7, color="#888")
        fig.suptitle(f"Position p = {p}: what the hidden state h{p} is trained to predict", fontsize=11)
        plt.tight_layout(rect=(0, 0, 1, .95))
        frames.append(_frame_to_pil(fig)); plt.close(fig)
    return save_gif(frames, path, 1100)


# ----------------------------------------------------------------------------- GIF 2: self-speculative decoding (illustrative)
def specdec_gif(path):
    """Illustrative, scripted comparison of tokens produced per verification pass. NOT measured:
    the acceptance pattern is fixed by hand to show the mechanism described in Gloeckle et al. (2024) and Teoh et al. (2025)."""
    target = list("the cat sat on the mat and purrs")
    n = len(target)
    # script: list of (drafted_count, accepted_count) per forward pass
    scripts = {
        "Next-token (no drafting)": [(1, 1)] * n,
        "MTP, fixed draft d=2": [(2, 2), (2, 1), (2, 2), (2, 1), (2, 2), (2, 2), (2, 1), (2, 1), (2, 2)],
        "NextLat, variable draft": [(5, 4), (4, 4), (3, 3), (4, 3)],
    }
    # make sure scripts cover the text exactly (accepted+1 correction token counts as emitted)
    states = {}
    for name, s in scripts.items():
        pos, hist = 0, []
        for dr, ac in s:
            if pos >= n: break
            ac = min(ac, n - pos)
            hist.append((pos, min(dr, n - pos), ac)); pos += ac + (1 if ac < dr else 0) if name != "Next-token (no drafting)" else ac
        states[name] = hist
    maxf = max(len(h) for h in states.values())
    frames = []
    for f in range(maxf + 1):
        fig, axs = plt.subplots(3, 1, figsize=(10, 4.6))
        for ax, (name, hist) in zip(axs, states.items()):
            ax.set_xlim(-.5, n + .5); ax.set_ylim(-.9, 1.2); ax.axis("off")
            k = min(f, len(hist))
            ax.set_title(f"{name}:  {k} forward pass{'es' if k != 1 else ''}", loc="left", fontsize=10)
            done = 0
            for (pos, dr, ac) in hist[:k]:
                done = pos + ac + (1 if ac < dr and name != "Next-token (no drafting)" else 0)
            for i, ch in enumerate(target):
                col = "#F4F4F4"
                if i < done: col = "#BFE3BF"
                _box(ax, i, 0.3, .8, .7, ch if ch != " " else "·", col, fs=9)
            if 0 < k <= len(hist):
                pos, dr, ac = hist[k - 1]
                for i in range(pos, min(pos + dr, n)):
                    ok = i < pos + ac
                    _box(ax, i, -.55, .8, .45, "✓" if ok else "✗", "#D6F0D6" if ok else "#F6C5C5", fs=8)
        fig.suptitle("Illustrative (scripted acceptance pattern, not measured): tokens emitted per verification pass", fontsize=10)
        plt.tight_layout(rect=(0, 0, 1, .93)); frames.append(_frame_to_pil(fig)); plt.close(fig)
    return save_gif(frames, path, 1200), {k: len(v) for k, v in states.items()}


# ----------------------------------------------------------------------------- GIF 3: training curves from real runs
def training_gif(hist: dict, path, every=1):
    """hist[name] = (steps list, solve-rate list). Animates the measured curves."""
    colors = {"NTP": C["ntp"], "MTP": C["mtp"], "JTP": C["jtp"], "NextLat": C["nl"]}
    maxn = max(len(v[0]) for v in hist.values())
    frames = []
    for k in list(range(1, maxn + 1, every)) + [maxn]:
        fig, ax = plt.subplots(figsize=(7, 3.8))
        for name, (st, sr) in hist.items():
            kk = min(k, len(st))
            ax.plot(st[:kk], np.array(sr[:kk]) * 100, "-o", ms=3, color=colors.get(name, None), label=name)
        allst = max(max(v[0]) for v in hist.values())
        ax.set_xlim(0, allst); ax.set_ylim(0, 105); ax.axhline(50, color="k", ls=":", lw=1); ax.text(allst * .01, 52, "chance on G(2,l)", fontsize=8)
        ax.set_xlabel("training step"); ax.set_ylabel("exact-path solve rate (%)"); ax.legend(loc="center right", fontsize=8); ax.set_title("Star graph: measured solve rate during training")
        plt.tight_layout(); frames.append(_frame_to_pil(fig)); plt.close(fig)
    return save_gif(frames, path, 350)


def star_graph_figure(ax, d=2, l=5, pick=0, title=None):
    """Draw G(d,l): centre node and d branches of l-1 further nodes; highlight the branch containing the goal."""
    ax.set_aspect("equal"); ax.axis("off")
    ang = [np.pi * 2 * k / d for k in range(d)] if d > 1 else [0]
    ax.add_patch(plt.Circle((0, 0), .22, fc="#FFD58A", ec="k", zorder=3)); ax.text(0, 0, "n1", ha="center", va="center", fontsize=8, zorder=4)
    for b, a in enumerate(ang):
        prev = (0, 0)
        for j in range(1, l):
            pt = (j * .8 * np.cos(a + .0), j * .8 * np.sin(a))
            good = b == pick
            ax.plot([prev[0], pt[0]], [prev[1], pt[1]], color="#2CA02C" if good else "#999", lw=2.5 if good else 1.5, zorder=1)
            ax.add_patch(plt.Circle(pt, .22, fc="#BFE3BF" if good else "#EEE", ec="k", zorder=3))
            ax.text(*pt, f"n{j + 1}" if good else "?", ha="center", va="center", fontsize=8, zorder=4)
            prev = pt
    ax.set_xlim(-l * .9, l * .9); ax.set_ylim(-l * .5 if d > 1 else -1, l * .5 if d > 1 else 1)
    if title: ax.set_title(title, fontsize=10)
