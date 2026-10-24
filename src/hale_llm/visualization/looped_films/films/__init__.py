"""One module per looped-transformer film; importing registers them in ``LOOP_FILMS`` (in order)."""

from hale_llm.visualization.looped_films.films.act_film import ACTFilm
from hale_llm.visualization.looped_films.films.fixed_point_film import FixedPointFilm
from hale_llm.visualization.looped_films.films.inside_a_pass_film import InsideAPassFilm
from hale_llm.visualization.looped_films.films.mor_film import MoRFilm
from hale_llm.visualization.looped_films.films.recurrent_depth_film import RecurrentDepthFilm
from hale_llm.visualization.looped_films.films.relaxed_film import RelaxedFilm
from hale_llm.visualization.looped_films.films.scratchpad_film import ScratchpadFilm
from hale_llm.visualization.looped_films.films.sharing_film import SharingFilm
from hale_llm.visualization.looped_films.films.unrolling_film import UnrollingFilm

__all__ = [
    "ACTFilm",
    "FixedPointFilm",
    "InsideAPassFilm",
    "MoRFilm",
    "RecurrentDepthFilm",
    "RelaxedFilm",
    "ScratchpadFilm",
    "SharingFilm",
    "UnrollingFilm",
]
