"""``LoopFilm``: base class and registry of the looped-transformer concept animations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import ClassVar

from hale_core.registry import NamedRegistry

from hale_llm.visualization.explainers.canvas import TEAL, GifWriter, Stage
from hale_llm.visualization.looped_films.loop_lab import LoopLab

#: film name -> :class:`LoopFilm` subclass (one module per film under ``films/``).
LOOP_FILMS = NamedRegistry("loop_film", scope="hale_llm")


class LoopFilm(ABC):
    """One concept of looped transformers as a short animated GIF.

    The films share the explainer films' visual grammar (dark stage, header + phase pill, one
    short headline per scene). Films that show model behaviour draw it from a toy model trained
    by :class:`LoopLab`; purely structural films (weight sharing schedules ...) need no model.

    Attributes:
        name: registry key and file stem (``assets/looped/<name>.gif``).
        title: headline of every frame.
        subtitle: one-line summary.
    """

    name: ClassVar[str]
    title: ClassVar[str]
    subtitle: ClassVar[str]

    def stage(self, phase: str, caption: str = "", color: str = TEAL) -> Stage:
        """A new frame with this film's title and subtitle."""
        return Stage(self.title, self.subtitle, phase, caption, phase_color=color)

    @abstractmethod
    def render(self, lab: LoopLab, writer: GifWriter) -> None:
        """Append every frame of the film to ``writer``."""

    def build(self, path: str | Path, lab: LoopLab | None = None) -> Path:
        """Render the film to ``path`` (a fresh :class:`LoopLab` trains what it needs)."""
        writer = GifWriter()
        self.render(lab or LoopLab(), writer)
        return writer.save(path)


def register_loop_film(cls: type[LoopFilm]) -> type[LoopFilm]:
    """Register ``cls`` under :attr:`LoopFilm.name` (called at the bottom of each film module)."""
    return LOOP_FILMS.add(cls.name, cls)


def build_loop_films(
    out_dir: str | Path = "assets/looped", names=None, lab: LoopLab | None = None
) -> list[Path]:
    """Render the films ``names`` (default: all, in registry order) into ``out_dir/<name>.gif``."""
    lab = lab or LoopLab()
    return [
        LOOP_FILMS.get(n)().build(Path(out_dir) / f"{n}.gif", lab)
        for n in names or list(LOOP_FILMS.keys())
    ]
