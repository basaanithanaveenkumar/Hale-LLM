"""Shared helpers for the RL-for-LLMs/VLMs notebooks.

* :mod:`rlkit.viz`     - one plotting style, GIF :class:`~rlkit.viz.Animator`, diagram boxes/arrows
* :mod:`rlkit.toy_lm`  - :class:`~rlkit.toy_lm.TinyGPT`, ``generate`` / ``sequence_logprobs``, toy tasks
* :mod:`rlkit.toy_vlm` - :class:`~rlkit.toy_vlm.TinyVLM` + synthetic ShapeWorld VQA

The RL *algorithms* are never in here: each notebook writes its algorithm from scratch, so the
reader sees every line. This package only holds the plumbing that would otherwise be copy-pasted.
"""

from .viz import COLORS, Animator, plot_curves, setup_style, show_frame, show_gif, smooth

__all__ = ["COLORS", "Animator", "plot_curves", "setup_style", "show_frame", "show_gif", "smooth"]
