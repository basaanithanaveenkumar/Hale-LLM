"""Dion (Microsoft) for the hidden weight matrices, AdamW for everything else."""

from __future__ import annotations

from typing import Any

import torch._dynamo
from loguru import logger

from hale_llm.optim.base_optimizer import BaseOptimizer
from hale_llm.optim.parameter_groups import split_parameters
from hale_llm.optim.registry import register_optimizer


# dion compiles its AdamW update with ``torch.compile(fullgraph=True)``. PyTorch allows 8 recompiles per compiled
# function (one per new parameter shape or rank), and with ``fullgraph`` the 9th is an error that stops training, not
# a fallback to eager. The AdamW group of a real model (norm gains, biases, embeddings, routers, MTP and MoE layers)
# has many distinct shapes, so the limit is raised to this.
RECOMPILE_LIMIT = 64


@register_optimizer("dion")
class DionOptimizer(BaseOptimizer):
    """Dion: orthonormal updates from a low-rank power iteration (Ahn et al., 2025).

    Like Muon, Dion orthonormalises the update of each *matrix*, so it takes 2-D weights only. Unlike Muon it first
    shrinks the matrix to ``rank`` columns (power iteration with error feedback), which is cheaper and, on many
    GPUs, much cheaper to communicate. This uses the single-device implementation of the ``dion`` package
    (``dion.DionSimple``), which keeps the AdamW updates for the other parameters in the same optimizer.

    Parameters are split as for Muon (see :func:`split_parameters`):

    * Dion: 2-D weights inside the transformer blocks (attention and feed-forward projections), learning rate
      ``dion_lr`` (the package scales it by ``sqrt(d_out / d_in)`` per matrix) and weight decay ``weight_decay``.
    * AdamW: everything else (token embeddings, output head, routers, norm gains, biases), learning rate ``lr``.
      Weight decay is 0 for these, the same convention as the default ``adamw`` optimizer.

    Config keys read: ``lr``, ``weight_decay``, ``adam_beta1``, ``adam_beta2``, ``dion_lr`` (default 0.02),
    ``dion_mu`` (momentum, default 0.95) and ``dion_rank`` (default 768, capped at the smaller side of each matrix).
    """

    #: The trainer passes ``model.named_parameters()`` to classes that set this.
    needs_named_parameters = True

    def _make_optimizer(self, params: Any, config: dict[str, Any]):
        try:
            from dion import DionSimple
        except ImportError as error:  # the package is a dependency, but fail with the fix, not a bare traceback
            raise ImportError(
                "optimizer_type 'dion' needs the dion package: "
                "pip install 'dion @ git+https://github.com/microsoft/dion.git'"
            ) from error

        torch._dynamo.config.recompile_limit = max(torch._dynamo.config.recompile_limit, RECOMPILE_LIMIT)
        dion_params, adamw_params = split_parameters(params)
        if not dion_params:
            raise ValueError("dion: model has no 2-D hidden weights to optimise with Dion")
        dion_lr = config.get("dion_lr", 0.02)
        rank = config.get("dion_rank", 768)
        logger.info(
            "dion: {:,} params on Dion (lr={}, rank<={}), {:,} params on AdamW (lr={})",
            sum(p.numel() for p in dion_params), dion_lr, rank,
            sum(p.numel() for p in adamw_params), config["lr"],
        )
        # AdamW first: the trainer logs param_groups[0]["lr"], which should be train.lr.
        groups = []
        if adamw_params:
            groups.append({"params": adamw_params, "algorithm": "adamw", "lr": config["lr"], "weight_decay": 0.0})
        groups.append({
            "params": dion_params, "algorithm": "dion", "lr": dion_lr,
            "mu": config.get("dion_mu", 0.95), "weight_decay": config.get("weight_decay", 0.0),
        })
        return DionSimple(
            groups,
            lr=config["lr"],
            betas=(config.get("adam_beta1", 0.9), config.get("adam_beta2", 0.95)),
            rank=rank,
        )
