"""AdamW with the settings usually used for language models."""

from __future__ import annotations

from typing import Any

import torch
from hale_llm.optim.base_optimizer import BaseOptimizer
from loguru import logger

from hale_llm.optim.parameter_groups import split_parameters
from hale_llm.optim.registry import register_optimizer


@register_optimizer("adamw")
class AdamWOptimizer(BaseOptimizer):
    """AdamW, betas (0.9, 0.95), weight decay 0.1, epsilon 1e-8, weight decay on the weight matrices only.

    Two changes from plain ``torch.optim.AdamW``, both standard for LLM training
    (GPT-3, Llama):

    * ``beta2 = 0.95`` instead of 0.999. A shorter second-moment memory reacts faster
      to the loss spikes that appear at large learning rates.
    * Weight decay is skipped for norm gains, biases and embeddings. Shrinking
      those toward zero hurts quality without regularising anything useful.

    Config keys read: ``lr``, ``weight_decay`` (0.1), ``adam_beta1`` (0.9), ``adam_beta2`` (0.95), ``adam_eps`` (1e-8).
    With the trainer's gradient clipping of 1.0 this is the "very vanilla" LLM recipe.
    """

    needs_named_parameters = True

    def _make_optimizer(self, params: Any, config: dict[str, Any]):
        decayed, undecayed = split_parameters(params)
        weight_decay = config.get("weight_decay", 0.1)
        betas = (config.get("adam_beta1", 0.9), config.get("adam_beta2", 0.95))
        eps = config.get("adam_eps", 1e-8)
        logger.info(
            "adamw: betas={} | eps {} | weight decay {} on {:,} matrix params, none on {:,} others",
            betas,
            eps,
            weight_decay,
            sum(p.numel() for p in decayed),
            sum(p.numel() for p in undecayed),
        )
        groups = [{"params": decayed, "weight_decay": weight_decay}]
        if undecayed:
            groups.append({"params": undecayed, "weight_decay": 0.0})
        return torch.optim.AdamW(groups, lr=config["lr"], betas=betas, eps=eps)
