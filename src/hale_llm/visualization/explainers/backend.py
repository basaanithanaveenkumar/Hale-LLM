"""Matplotlib backend choice shared by the explainer animation modules.

Headless rendering (Agg) for the CLI and scripts. Inside a Jupyter kernel the notebook's own
backend is kept: switching to Agg there would silently stop every inline plot from being shown.
Import ``plt`` from here instead of ``matplotlib.pyplot`` so the choice is made exactly once.
"""

from __future__ import annotations

import sys

import matplotlib

if "ipykernel" not in sys.modules:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402 - must follow the backend choice

__all__ = ["plt"]
