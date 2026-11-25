"""Load and inherit YAML into a validated RunConfig."""

from pathlib import Path

import yaml
from loguru import logger

from hale_llm.config.merge import deep_merge
from hale_llm.config.run import RunConfig


def _read_with_parents(path: Path, seen: tuple[Path, ...] = ()) -> dict:
    """Read a YAML file and merge it over its ``inherits`` chain (any depth, no cycles).

    Example: ``ablation.yaml`` inherits ``recipe.yaml``, which inherits ``small.yaml``, which
    inherits ``base.yaml``. Values in a child replace the parent's; dicts are merged key by key.
    """
    if path in seen:
        chain = " -> ".join(str(p) for p in (*seen, path))
        raise ValueError(f"config inheritance cycle: {chain}")
    raw = yaml.safe_load(path.read_text()) or {}
    parent_name = raw.pop("inherits", None)
    if parent_name is None:
        return raw
    parent = (path.parent / parent_name).resolve()
    if not parent.exists():
        raise FileNotFoundError(f"{path} inherits {parent_name}, which does not exist")
    return deep_merge(_read_with_parents(parent, (*seen, path)), raw)


def load_config(path: str | Path) -> RunConfig:
    path = Path(path).resolve()
    if not path.exists():
        logger.error("config file not found: {}", path)
        raise FileNotFoundError(path)
    raw = _read_with_parents(path)
    try:
        cfg = RunConfig.model_validate(raw)
    except Exception:
        logger.exception("invalid config {}", path)
        raise
    cfg.experiment.source_yaml = str(path)
    logger.info("loaded config variant={} from {}", cfg.variant, path)
    return cfg
