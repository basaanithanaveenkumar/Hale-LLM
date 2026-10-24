"""Animated films of how looped transformers work, one per concept and variant.

``hale-llm-loop-films`` (or :func:`build_loop_films`) renders them into ``assets/looped/``:

* ``unrolling`` - stacked (6 blocks) vs looped (1 block × 6); structural.
* ``inside_a_pass`` - logit lens: next-character guesses sharpen pass by pass (trained UT toy).
* ``act_halting`` - per-token halting rings, tokens stopping at different passes (UT + ACT toy).
* ``mixture_of_recursions`` - the router grants depths, tokens exit the loop (MoR toy).
* ``fixed_point`` - a state spiralling into ``h* = f(h*)``, with vs without post-norms (DEQ toys).
* ``recurrent_depth`` - random-T training and test-time loop scaling (docstring char LMs).
* ``relaxed_recursion`` - shared W + per-pass LoRA deltas as heat maps (relaxed toy).
* ``sharing_patterns`` - cycle / cycle_rev / sequence schedules of a 2-layer core; structural.
* ``scratchpad`` - memory slots and who may read whom under a causal mask; structural.

:class:`LoopLab` trains and traces the toy models (also used by ``notebooks/looped/``).
"""

from hale_llm.visualization.looped_films import films as _films  # noqa: F401  (registers films)
from hale_llm.visualization.looped_films.docstring_corpus import docstring_corpus
from hale_llm.visualization.looped_films.loop_film import (
    LOOP_FILMS,
    LoopFilm,
    build_loop_films,
    register_loop_film,
)
from hale_llm.visualization.looped_films.loop_lab import LoopLab
from hale_llm.visualization.looped_films.pass_trace import PassTrace

__all__ = [
    "LOOP_FILMS",
    "LoopFilm",
    "LoopLab",
    "PassTrace",
    "build_loop_films",
    "docstring_corpus",
    "register_loop_film",
]
