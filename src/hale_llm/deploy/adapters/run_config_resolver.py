"""Locate and load a HaloDiffusionLLM run config (with optional overrides) for deployment."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from hale_core.deploy.exceptions import ConfigError
from loguru import logger

#: Repository root when installed in editable / source mode (``src/hale_llm/deploy/adapters``).
REPO_ROOT = Path(__file__).resolve().parents[4]


class RunConfigResolver:
    """Resolve a run-config path and apply ``overrides`` on top of the inherited YAML.

    Paths are tried as given (absolute or relative to the working directory) and then
    relative to the repository root, so deployment configs can write
    ``config: configs/diffusion/nano.yaml`` and run from anywhere inside or outside the repo.

    Args:
        path: run-config YAML (training config of the model being deployed).
        overrides: nested dict deep-merged onto the loaded config (e.g. a tiny ``model``
            section for CI, or ``model.loop.n_loops`` for a different compute depth).
    """

    def __init__(self, path: str, overrides: Mapping[str, Any] | None = None) -> None:
        self.path = self.locate(path)
        self.overrides = dict(overrides or {})

    @staticmethod
    def locate(path: str) -> Path:
        """Resolve ``path`` relative to the cwd, then the repo root.

        Raises:
            hale_core.exceptions.ConfigError: if neither exists.
        """
        for candidate in (Path(path).expanduser(), REPO_ROOT / path):
            if candidate.is_file():
                return candidate.resolve()
        raise ConfigError(f"run config {path!r} not found (cwd or {REPO_ROOT})")

    def load(self):
        """Return the validated :class:`hale_llm.config.run.RunConfig`."""
        from hale_llm.config.load import load_config
        from hale_llm.config.merge import deep_merge
        from hale_llm.config.run import RunConfig

        cfg = load_config(self.path)
        if not self.overrides:
            return cfg
        try:
            merged = RunConfig.model_validate(deep_merge(cfg.model_dump(), self.overrides))
        except Exception as exc:  # pydantic ValidationError and friends
            raise ConfigError(f"invalid overrides {self.overrides} for {self.path}: {exc}") from exc
        logger.debug("run config {} with overrides {}", self.path.name, sorted(self.overrides))
        return merged
