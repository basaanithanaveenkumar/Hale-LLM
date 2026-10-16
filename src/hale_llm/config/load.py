"""Load a YAML run config (with its inheritance chain) into a validated :class:`RunConfig`."""

from __future__ import annotations

from pathlib import Path

from hale_core.config.config_loader import ConfigLoader
from loguru import logger
from pydantic import ValidationError

from hale_llm.config.run import RunConfig


def load_config(path: str | Path) -> RunConfig:
    """Read ``path``, merge its ``inherits:`` / ``_base_`` ancestors and validate the result.

    Inheritance is resolved by HaleBlocks' :class:`~hale_core.config.config_loader.ConfigLoader`,
    so chains may be any depth, a file may list several bases (later ones win), ``_delete_: true``
    replaces a mapping instead of merging into it, and cycles raise a clear error.

    Args:
        path: the run config, e.g. ``configs/diffusion/small.yaml``.

    Returns:
        The validated config, with ``experiment.source_yaml`` set to the resolved path.

    Raises:
        FileNotFoundError: if ``path`` does not exist.
        hale_core.exceptions.ConfigError: if a base file is missing, unreadable or cyclic.
        pydantic.ValidationError: if the merged YAML does not match :class:`RunConfig`.

    Example:
        >>> cfg = load_config("configs/diffusion/small.yaml")  # doctest: +SKIP
        >>> cfg.variant
        'diffusion'
    """
    path = Path(path).resolve()
    logger.debug("loading config from {}", path)
    if not path.exists():
        logger.error("config file not found: {}", path)
        raise FileNotFoundError(path)
    raw = ConfigLoader().load(path)
    try:
        cfg = RunConfig.model_validate(raw)
    except ValidationError:
        logger.error("invalid config {}", path)
        raise
    cfg.experiment.source_yaml = str(path)
    logger.info("loaded config variant={} from {}", cfg.variant, path)
    return cfg
