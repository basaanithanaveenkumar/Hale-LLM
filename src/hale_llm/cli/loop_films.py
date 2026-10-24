"""``hale-llm-loop-films``: render the looped-transformer concept animations into a folder."""

from __future__ import annotations

import argparse
import time

from loguru import logger

from hale_llm.utils.logging import setup_logging


def main() -> None:
    """Parse arguments and render the requested films (default: all into ``assets/looped``)."""
    from hale_llm.visualization.looped_films import LOOP_FILMS, LoopLab, build_loop_films

    names = list(LOOP_FILMS.keys())
    p = argparse.ArgumentParser(prog="hale-llm-loop-films", description=__doc__)
    p.add_argument("--only", nargs="*", choices=names, help="subset of films (default: all)")
    p.add_argument("--out", default="assets/looped", help="output directory for <film>.gif")
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = p.parse_args()
    setup_logging(level=args.log_level, log_file=None, force=True)
    lab = LoopLab()  # shared: films reuse each other's trained toys
    for name in args.only or names:
        start = time.time()
        (path,) = build_loop_films(args.out, [name], lab)
        logger.info(
            "{} -> {} ({:.0f} KB, {:.0f}s)",
            name,
            path,
            path.stat().st_size / 1024,
            time.time() - start,
        )
        print(path)


if __name__ == "__main__":
    main()
