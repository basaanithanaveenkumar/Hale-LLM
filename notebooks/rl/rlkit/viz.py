"""Shared plotting + animation helpers so every notebook has the same visual grammar.

Visual grammar (keep it consistent across notebooks):

* ``COLORS["policy"]`` (blue)   - the model being trained / the policy
* ``COLORS["reference"]`` (gray) - the frozen reference / old policy
* ``COLORS["reward"]`` (green)  - rewards, good samples, "chosen" answers
* ``COLORS["penalty"]`` (red)   - penalties, bad samples, "rejected" answers
* ``COLORS["value"]`` (orange)  - critics, value functions, baselines
* ``COLORS["accent"]`` (purple) - anything else that needs highlighting (KL, clip ranges, ...)

GIFs are written with matplotlib's Pillow writer, so no ffmpeg is needed. A notebook builds a GIF by
writing one ``draw(ax_or_fig, frame_index)`` function and handing it to :class:`Animator`; the same
function can render a static frame with :func:`show_frame`, so the static picture and the animation
can never disagree.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional, Sequence, Tuple, Union

import matplotlib

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

COLORS = {
    "policy": "#2563eb",
    "reference": "#6b7280",
    "reward": "#16a34a",
    "penalty": "#dc2626",
    "value": "#ea580c",
    "accent": "#7c3aed",
    "light": "#e5e7eb",
    "ink": "#111827",
}

# notebooks/rl/rlkit/viz.py -> <repo>/assets/rl/course (the repo keeps generated media under assets/).
ASSETS_DIR = Path(__file__).resolve().parents[3] / "assets" / "rl" / "course"


def setup_style() -> None:
    """Apply one clean, readable style to every figure (call once per notebook)."""
    plt.rcParams.update({
        "figure.dpi": 100,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.titleweight": "bold",
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "legend.frameon": False,
        "font.size": 10,
    })


def asset_path(name: str) -> Path:
    """Absolute path for a generated asset (``assets/rl/course/<name>``), creating the folder if needed."""
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    return ASSETS_DIR / name


def smooth(values: Sequence[float], window: int = 20) -> list:
    """Trailing moving average used for noisy learning curves (window shrinks at the start)."""
    out, running = [], 0.0
    values = list(values)
    for i, v in enumerate(values):
        running += v
        if i >= window:
            running -= values[i - window]
        out.append(running / min(i + 1, window))
    return out


def plot_curves(curves: dict, title: str = "", xlabel: str = "step", ylabel: str = "",
                window: int = 1, ax=None, colors: Optional[dict] = None):
    """Plot several named learning curves on one axis (optionally smoothed).

    ``curves`` maps a label to a list of values. ``colors`` optionally maps the same labels to colours;
    otherwise matplotlib's cycle is used.
    """
    own = ax is None
    if own:
        _, ax = plt.subplots(figsize=(7, 3.8))
    for label, values in curves.items():
        ys = smooth(values, window) if window > 1 else list(values)
        ax.plot(range(len(ys)), ys, label=label, lw=2, color=(colors or {}).get(label))
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if len(curves) > 1:
        ax.legend()
    if own:
        plt.tight_layout()
        plt.show()
    return ax


DrawFn = Callable[[plt.Figure, int], None]


class Animator:
    """Render ``draw(fig, i)`` for ``i in range(n_frames)`` into a GIF.

    ``draw`` receives the figure (clear it with ``fig.clf()`` or reuse axes) and the frame index. Each
    frame is drawn from scratch, so the drawing function is a pure function of ``i`` and can be
    reused for a static preview via :func:`show_frame`.

    Example::

        def draw(fig, i):
            fig.clf(); ax = fig.add_subplot(); ax.bar(range(3), probs_history[i])
        Animator(draw, n_frames=len(probs_history), figsize=(5, 3)).save("bandit.gif")
    """

    def __init__(self, draw: DrawFn, n_frames: int, figsize: Tuple[float, float] = (6, 4),
                 fps: int = 6, dpi: int = 80):
        self.draw = draw
        self.n_frames = n_frames
        self.figsize = figsize
        self.fps = fps
        self.dpi = dpi

    def save(self, name: str, display: bool = True) -> Path:
        """Write ``assets/<name>`` and (optionally) display it inline in the notebook."""
        path = asset_path(name)
        fig = plt.figure(figsize=self.figsize)
        anim = FuncAnimation(fig, lambda i: self.draw(fig, i), frames=self.n_frames, repeat=False)
        anim.save(str(path), writer=PillowWriter(fps=self.fps), dpi=self.dpi)
        plt.close(fig)
        if display:
            show_gif(name)
        return path


def show_frame(draw: DrawFn, i: int, figsize: Tuple[float, float] = (6, 4)) -> None:
    """Render a single frame of an animation as a normal inline figure."""
    fig = plt.figure(figsize=figsize)
    draw(fig, i)
    plt.show()


def show_gif(name: str, width: int = 640) -> None:
    """Embed ``assets/<name>`` inline (base64, so the executed notebook is self-contained)."""
    import base64

    from IPython.display import HTML, display

    data = base64.b64encode(asset_path(name).read_bytes()).decode("ascii")
    display(HTML(f'<img src="data:image/gif;base64,{data}" width="{width}">'))


def save_figure(fig: plt.Figure, name: str) -> Path:
    """Save a static figure into ``assets/`` (PNG) - handy for README thumbnails."""
    path = asset_path(name)
    fig.savefig(path, bbox_inches="tight", dpi=110)
    return path


def draw_box(ax, xy: Tuple[float, float], w: float, h: float, text: str, color: str,
             fontsize: int = 10, alpha: float = 0.18) -> None:
    """A rounded labelled box - the building block of the hand-drawn pipeline diagrams."""
    from matplotlib.patches import FancyBboxPatch

    ax.add_patch(FancyBboxPatch(xy, w, h, boxstyle="round,pad=0.02,rounding_size=0.05",
                                fc=matplotlib.colors.to_rgba(color, alpha), ec=color, lw=2))
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center", fontsize=fontsize,
            color=COLORS["ink"])


def draw_arrow(ax, start: Tuple[float, float], end: Tuple[float, float], color: str = "#111827",
               text: Union[str, None] = None, lw: float = 1.8) -> None:
    """Arrow between two points with an optional label at its midpoint."""
    ax.annotate("", xy=end, xytext=start, arrowprops=dict(arrowstyle="-|>", color=color, lw=lw))
    if text:
        ax.text((start[0] + end[0]) / 2, (start[1] + end[1]) / 2 + 0.03, text, ha="center",
                fontsize=9, color=color)
