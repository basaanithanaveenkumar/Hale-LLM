"""Relaxed recursion: one shared weight W, plus a small learned LoRA delta per pass."""

from __future__ import annotations

import torch
from hale_core.nn.layers.loop import LoopLoRALinear

from hale_llm.visualization.explainers.canvas import (
    CYAN,
    FAINT,
    GOLD,
    INK,
    MUTED,
    PANEL,
    GifWriter,
    ease,
)
from hale_llm.visualization.looped_films.loop_film import LoopFilm, register_loop_film
from hale_llm.visualization.looped_films.loop_lab import LoopLab

CROP = 32  # show the top-left 32 x 32 block of the matrices


class RelaxedFilm(LoopFilm):
    """Heat maps of a trained relaxed-recursive toy: ``W``, ``ΔW_t = B_t A_t`` and ``W + ΔW_t``.

    The deltas are rank 4 (visible as banded structure) and much smaller than ``W``; each pass gets
    its own, so the loop is no longer forced to apply the identical map at every step.
    """

    name = "relaxed_recursion"
    title = "Relaxed recursion: shared W + a tiny per-pass delta"
    subtitle = (
        "W_t = W + B_t A_t  (rank 4, B_t starts at 0: training starts from exact weight tying)"
    )

    def render(self, lab: LoopLab, writer: GifWriter) -> None:
        """Draw the film into ``writer`` from a trained ``relaxed_recursive`` toy model."""
        run = lab.train("relaxed_recursive")
        layer = next(m for m in lab.stack(run).core.modules() if isinstance(m, LoopLoRALinear))
        w = layer.weight.detach()[:CROP, :CROP]
        deltas = [layer.delta_weight(t).detach()[:CROP, :CROP] for t in range(layer.down.shape[0])]
        scale_w = float(w.abs().max())
        scale_d = max(float(d.abs().max()) for d in deltas) or 1.0
        for t, delta in enumerate(deltas):
            ratio = float(layer.delta_weight(t).detach().norm() / layer.weight.detach().norm())
            for g in range(6):
                u = ease((g + 1) / 6)
                s = self.stage(
                    f"pass {t + 1} / {len(deltas)}",
                    f"pass {t + 1} uses W + ΔW_{t + 1}  (|ΔW| = {ratio:.1%} of |W|)",
                    CYAN,
                )
                self._heat(s, w, scale_w, (4.0, 8.0, 30.0, 38.0), "shared W (same every pass)")
                s.text(33.0, 23.0, "+", size=26, color=INK, weight="bold")
                self._heat(
                    s,
                    delta * u,
                    scale_d,
                    (37.0, 8.0, 63.0, 38.0),
                    f"ΔW_{t + 1} = B_{t + 1} A_{t + 1}  (×{scale_w / scale_d:.0f} colour)",
                )
                s.text(66.0, 23.0, "=", size=26, color=INK, weight="bold")
                self._heat(
                    s,
                    w + delta * u,
                    scale_w,
                    (70.0, 8.0, 96.0, 38.0),
                    f"W_{t + 1} used in pass {t + 1}",
                )
                s.footer(
                    f"one FFN projection of the trained toy (top-left {CROP}×{CROP}) · deltas add "
                    f"{layer.down.numel() + layer.up.numel():,} parameters to this layer's "
                    f"{layer.weight.numel():,}"
                )
                s.progress((t + u) / len(deltas), GOLD)
                writer.stage(s, 85)
            writer.hold(600)
        writer.hold(2400)

    @staticmethod
    def _heat(s, m: torch.Tensor, scale: float, box, label: str) -> None:
        x0, y0, x1, y1 = box
        s.box(x0 - 0.6, y0 - 0.6, x1 + 0.6, y1 + 0.6, face=PANEL, edge=FAINT, radius=0.8)
        s.ax.imshow(
            m.float().numpy(),
            cmap="RdBu_r",
            vmin=-scale,
            vmax=scale,
            extent=(x0, x1, y0, y1),
            interpolation="nearest",
            zorder=3,
            aspect="auto",
        )
        s.text((x0 + x1) / 2, y1 + 2.2, label, size=9, color=MUTED)


register_loop_film(RelaxedFilm)
