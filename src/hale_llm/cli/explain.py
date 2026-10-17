"""``hale-llm-explain``: render the animated explainer GIF of every (or some) paradigm(s)."""

from __future__ import annotations

import argparse
import time

from loguru import logger

from hale_llm.utils.logging import setup_logging


def main() -> None:
    """Render the explainer GIF of each requested paradigm; print one output path per line.

    Each paradigm trains its toy model, traces generation and writes ``<out>/<variant>.gif``.
    """
    from hale_llm.visualization.explainers import ORDER, build_explainer_gifs

    p = argparse.ArgumentParser(prog="hale-llm-explain", description=__doc__)
    p.add_argument("--variants", default="all", help=f"comma-separated subset of {', '.join(ORDER)}")
    p.add_argument("--out", default="assets/explainers", help="output directory for <variant>.gif")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = p.parse_args()
    setup_logging(level=args.log_level, log_file=None, force=True)
    variants = list(ORDER) if args.variants == "all" else [v.strip() for v in args.variants.split(",") if v.strip()]
    for name in variants:
        start = time.time()
        (path,) = build_explainer_gifs(args.out, [name], seed=args.seed)
        logger.info("{} -> {} ({:.0f} KB, {:.1f}s)", name, path, path.stat().st_size / 1024, time.time() - start)
        print(path)  # stdout = the command's product (one GIF path per line, pipeable)


if __name__ == "__main__":
    main()
