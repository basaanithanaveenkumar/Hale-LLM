"""``hale-llm-tokenizer-ablation``: compare tokenizer algorithms at the same vocabulary size.

Trains each registered tokenizer kind (default: ``bpe`` and ``superbpe``) on the run's
training documents with the same vocabulary size, and measures how well each compresses
held-out text. This is the cheap half of the BPE-versus-SuperBPE ablation; the expensive
half is training two models (see ``configs/ablations/`` and ``docs/tokenizer.md``).

Example::

    hale-llm-tokenizer-ablation --config configs/autoregressive/small.yaml --max-documents 20000
"""

from __future__ import annotations

import argparse
import json
import sys
import time

from loguru import logger

from hale_llm.config.schema import load_config
from hale_llm.tokenizer import TokenizerBuilder, TokenizerStore, bytes_per_token
from hale_llm.tokenizer.special_tokens import EOS_TOKEN
from hale_llm.tokenizer.training import TOKENIZER_TRAINERS, build_tokenizer_trainer
from hale_llm.utils.logging import setup_logging


def compare_kinds(cfg, kinds: list[str], vocab_size: int, train_docs, eval_docs) -> list[dict]:
    """Train every kind on ``train_docs`` and measure it on ``eval_docs``."""
    rows = []
    for kind in kinds:
        settings = cfg.tokenizer.model_copy(update={"kind": kind})
        started = time.time()
        raw = build_tokenizer_trainer(settings).train(train_docs, vocab_size, [EOS_TOKEN])
        seconds = time.time() - started
        tokenizer = TokenizerStore.wrap(raw, with_mask=False)
        ratio = bytes_per_token(tokenizer, eval_docs)
        rows.append(
            {
                "kind": kind,
                "vocab_size": len(tokenizer),
                "bytes_per_token": round(ratio, 4),
                "train_seconds": round(seconds, 1),
            }
        )
    # Tokens needed for the same text, relative to the first kind (the baseline).
    baseline = rows[0]["bytes_per_token"]
    for row in rows:
        row["tokens_vs_first"] = round(baseline / row["bytes_per_token"], 4)
    return rows


def format_table(rows: list[dict]) -> str:
    header = f"{'kind':<10} {'vocab':>8} {'bytes/token':>12} {'tokens vs first':>16} {'train s':>8}"
    lines = [header, "-" * len(header)]
    for r in rows:
        lines.append(
            f"{r['kind']:<10} {r['vocab_size']:>8,} {r['bytes_per_token']:>12.3f} "
            f"{r['tokens_vs_first']:>16.3f} {r['train_seconds']:>8.1f}"
        )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="hale-llm-tokenizer-ablation", description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--kinds", nargs="+", default=None, help="Default: every registered kind")
    parser.add_argument("--vocab-size", type=int, default=None, help="Default: from the model size")
    parser.add_argument("--max-documents", type=int, default=None, help="Train on at most this many documents")
    parser.add_argument("--json", default=None, help="Also write the results to this file")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    try:
        cfg = load_config(args.config)
        setup_logging(level=cfg.logging.level, log_file=None, force=True)
        builder = TokenizerBuilder(cfg, with_mask=False)
        vocab_size = args.vocab_size or builder.resolve_vocab_size()
        train_docs = list(builder.documents(cfg.data.train_split))
        if args.max_documents:
            train_docs = train_docs[: args.max_documents]
        eval_docs = list(builder.documents(cfg.data.val_split))
        kinds = args.kinds or sorted(TOKENIZER_TRAINERS)
        # `bpe` first so the "tokens vs first" column reads as "vs standard BPE".
        kinds.sort(key=lambda k: k != "bpe")

        rows = compare_kinds(cfg, kinds, vocab_size, train_docs, eval_docs)
        print(format_table(rows))
        if args.json:
            with open(args.json, "w") as handle:
                json.dump(rows, handle, indent=2)
    except Exception as err:
        logger.exception("hale-llm-tokenizer-ablation failed")
        sys.exit(f"hale-llm-tokenizer-ablation: {err}")


if __name__ == "__main__":
    main()
