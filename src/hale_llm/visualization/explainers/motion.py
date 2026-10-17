"""Easing curves and motion paths used to tween objects between frames."""

from __future__ import annotations

import math


# ---------------------------------------------------------------------------- easing
def ease(t: float) -> float:
    """Smooth in-out (cubic), clamped to ``[0, 1]``."""
    t = min(max(t, 0.0), 1.0)
    return 4 * t**3 if t < 0.5 else 1 - (-2 * t + 2) ** 3 / 2


def ease_out_back(t: float) -> float:
    """Overshoot then settle - the 'pop' of a committed token."""
    t = min(max(t, 0.0), 1.0)
    c1, c3 = 1.70158, 2.70158
    return 1 + c3 * (t - 1) ** 3 + c1 * (t - 1) ** 2


def pop(t: float, amount: float = 0.3) -> float:
    """Scale of a tile that pops in: grows past 1 by ``amount`` and settles back."""
    return 1.0 + amount * math.sin(math.pi * min(max(t, 0.0), 1.0))


def lerp(a: float, b: float, t: float) -> float:
    """Linear interpolation from ``a`` to ``b`` at ``t``."""
    return a + (b - a) * t


def lerp2(p: tuple[float, float], q: tuple[float, float], t: float) -> tuple[float, float]:
    """Linear interpolation between two points."""
    return lerp(p[0], q[0], t), lerp(p[1], q[1], t)


def arc_path(
    p: tuple[float, float], q: tuple[float, float], t: float, lift: float = 6.0
) -> tuple[float, float]:
    """Point at ``t`` on a parabolic flight from ``p`` to ``q`` (bulging up by ``lift``)."""
    x, y = lerp2(p, q, t)
    return x, y + lift * 4 * t * (1 - t)


def bezier(
    p: tuple[float, float], c: tuple[float, float], q: tuple[float, float], t: float
) -> tuple[float, float]:
    """Point at ``t`` on the quadratic Bézier ``p → q`` with control point ``c``."""
    a, b = (1 - t) ** 2, 2 * (1 - t) * t
    return a * p[0] + b * c[0] + t * t * q[0], a * p[1] + b * c[1] + t * t * q[1]


__all__ = ["ease", "ease_out_back", "pop", "lerp", "lerp2", "arc_path", "bezier"]
