"""Animations for the online RL notebook (``notebooks/rl/01_online_rl.ipynb``).

Every film renders a real :class:`hale_llm.rl.RLHistory` or a real model, so the GIFs show what
the code did, not an illustration of it.
"""

from hale_llm.visualization.rl_films.diffusion_log_prob_film import DiffusionLogProbFilm
from hale_llm.visualization.rl_films.grpo_step_film import GRPOStepFilm
from hale_llm.visualization.rl_films.masked_view import MaskedView
from hale_llm.visualization.rl_films.policy_evolution_film import PolicyEvolutionFilm
from hale_llm.visualization.rl_films.rl_film import RLFilm

__all__ = ["DiffusionLogProbFilm", "GRPOStepFilm", "MaskedView", "PolicyEvolutionFilm", "RLFilm"]
