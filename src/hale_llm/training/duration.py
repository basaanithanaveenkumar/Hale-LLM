"""Human-readable durations for the training logs."""

from __future__ import annotations


def format_duration(seconds: float) -> str:
    """``3725.4`` -> ``'1 h 02 min 05 s'``; shorter spans drop the leading units (``'4 min 07 s'``, ``'12.3 s'``)."""
    if seconds < 0:
        raise ValueError(f"a duration cannot be negative: {seconds}")
    if seconds < 60:
        return f"{seconds:.1f} s"
    whole = int(round(seconds))
    hours, remainder = divmod(whole, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours} h {minutes:02d} min {secs:02d} s"
    return f"{minutes} min {secs:02d} s"
