#!/usr/bin/env python3
"""Profile HaloDiffusionLLM backbone: latency, FLOPs, and matmul count per layer.

Usage:
    python scripts/profile_model.py
    python scripts/profile_model.py --arch dit --d-model 512 --n-layers 12 --device cuda
    python scripts/profile_model.py --torch-profiler --trace out/trace.json
    python scripts/profile_model.py --sort flops --top 30 --json out/profile.json

Supported archs: gpt, lgt, dit
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
# HaleBlocks is a sibling package that provides the profiler
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "HaleBlocks" / "src"))

import torch
from hale_core.nn import DiTBackbone, GPTBackbone, LGTBackbone
from hale_core.profiling import ModelProfiler
from loguru import logger

# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------

def _base_kwargs(args) -> dict:
    return dict(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_ff,
        max_length=args.seq_len,
        dropout=0.0,
        attn_impl=args.attn_impl,
        ffn_type=args.ffn_type,
    )


def build_gpt(args):
    model = GPTBackbone(**_base_kwargs(args), attn_type="causal")
    dummy = torch.randint(0, args.vocab_size, (args.batch_size, args.seq_len))
    return model, dummy


def build_lgt(args):
    model = LGTBackbone(**_base_kwargs(args), attn_type="block_causal")
    dummy = torch.randint(0, args.vocab_size, (args.batch_size, args.seq_len))
    return model, dummy


def build_dit(args):
    model = DiTBackbone(**_base_kwargs(args), attn_type="bidirectional", use_time_cond=True)
    dummy = torch.randint(0, args.vocab_size, (args.batch_size, args.seq_len))
    return model, dummy


_BUILDERS = {"gpt": build_gpt, "lgt": build_lgt, "dit": build_dit}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="HaloDiffusionLLM profiler")
    p.add_argument("--arch",          default="gpt", choices=list(_BUILDERS))
    p.add_argument("--device",        default="cpu")
    p.add_argument("--batch-size",    type=int, default=2)
    p.add_argument("--seq-len",       type=int, default=128)
    p.add_argument("--vocab-size",    type=int, default=8192)
    p.add_argument("--d-model",       type=int, default=256)
    p.add_argument("--n-heads",       type=int, default=4)
    p.add_argument("--n-layers",      type=int, default=6)
    p.add_argument("--d-ff",          type=int, default=1024)
    p.add_argument("--attn-impl",     default="mha", choices=["mha", "gqa", "mqa"])
    p.add_argument("--ffn-type",      default="mlp", choices=["mlp", "geglu", "moe"])
    p.add_argument("--top",           type=int, default=40)
    p.add_argument("--sort",          default="latency", choices=["latency", "flops", "matmuls"])
    p.add_argument("--warmup",        type=int, default=2)
    p.add_argument("--torch-profiler", action="store_true")
    p.add_argument("--trace",         default=None)
    p.add_argument("--json",          default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    logger.info(
        "profiling arch={} device={} batch={} seq={} d_model={} n_layers={} attn={} ffn={}",
        args.arch, device, args.batch_size, args.seq_len, args.d_model, args.n_layers,
        args.attn_impl, args.ffn_type,
    )

    model, dummy = _BUILDERS[args.arch](args)
    model = model.to(device).eval()
    dummy = dummy.to(device)

    with torch.no_grad():
        for _ in range(args.warmup):
            _ = model(dummy)

    profiler = ModelProfiler(
        model,
        use_cuda_events=(device.type == "cuda"),
        use_torch_profiler=args.torch_profiler,
        torch_profiler_trace=args.trace,
    )

    with torch.no_grad(), profiler.profile():
        _ = model(dummy)

    report = profiler.report()
    # stdout = the profiling report (the script's product); status messages go to the log
    print("\n" + report.summary_table(top_k=args.top, sort_by=args.sort))

    if args.torch_profiler:
        print("\n── torch.profiler key averages ──")
        print(report.torch_key_averages())

    if args.json:
        import json
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.json, "w") as f:
            json.dump(report.to_dict(), f, indent=2)
        logger.info("profile JSON saved to {}", args.json)

    if args.trace:
        logger.info("Chrome trace saved to {}", args.trace)


if __name__ == "__main__":
    main()
