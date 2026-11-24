"""Muon for the hidden weight matrices, AdamW for everything else."""

from __future__ import annotations

from typing import Any

import torch
from hale_core.nn.optim.base import BaseOptimizer
from loguru import logger

from hale_llm.optim.parameter_groups import split_parameters
from hale_llm.optim.registry import register_optimizer

@register_optimizer("muon")
class MuonOptimizer(BaseOptimizer):
    """Muon (Jordan et al., 2024) as shipped in ``torch.optim.Muon``, wired for LLMs.

    Muon orthogonalises each update, which is only defined for a *matrix*, so it
    accepts 2-D weights only. A language model also has 1-D tensors (norm gains,
    biases) and lookup tables (token embeddings, tied output head). Following the
    usual recipe, parameters are split in two groups:

    * Muon: 2-D weights inside the transformer blocks (attention and FFN projections),
      learning rate ``muon_lr``.
    * AdamW: everything else, learning rate ``lr``.

    Config keys read: ``lr``, ``weight_decay``, ``muon_lr`` (default 0.02),
    ``muon_momentum`` (default 0.95).
    """

    #: The trainer passes ``model.named_parameters()`` to classes that set this.
    needs_named_parameters = True

    def _make_optimizer(self, params: Any, config: dict[str, Any]):
        muon_params, adamw_params = split_parameters(params)
        if not muon_params:
            raise ValueError("muon: model has no 2-D hidden weights to optimise with Muon")
        logger.info(
            "muon: {:,} params on Muon (lr={}), {:,} params on AdamW (lr={})",
            sum(p.numel() for p in muon_params),
            config.get("muon_lr", 0.02),
            sum(p.numel() for p in adamw_params),
            config["lr"],
        )
        weight_decay = config.get("weight_decay", 0.0)
        # AdamW first: the trainer logs param_groups[0]["lr"], which should be train.lr.
        self._adamw = (
            torch.optim.AdamW(adamw_params, lr=config["lr"], weight_decay=weight_decay)
            if adamw_params
            else None
        )
        self._muon = torch.optim.Muon(
            muon_params,
            lr=config.get("muon_lr", 0.02),
            weight_decay=weight_decay,
            momentum=config.get("muon_momentum", 0.95),
            # Rescale updates so the learning rate transfers from AdamW-style settings.
            adjust_lr_fn="match_rms_adamw",
        )
        return self._muon

    # ----- two inner optimizers behave as one -----

    def _inner(self) -> list[torch.optim.Optimizer]:
        return [o for o in (self._adamw, self._muon) if o is not None]

    def zero_grad(self, set_to_none: bool = False) -> None:
        for opt in self._inner():
            opt.zero_grad(set_to_none=set_to_none)

    def step(self) -> None:
        for opt in self._inner():
            opt.step()

    def state_dict(self) -> dict[str, Any]:
        return {
            "adamw": self._adamw.state_dict() if self._adamw else None,
            "muon": self._muon.state_dict(),
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        if self._adamw is not None and state_dict.get("adamw") is not None:
            self._adamw.load_state_dict(state_dict["adamw"])
        self._muon.load_state_dict(state_dict["muon"])

    @property
    def param_groups(self):
        return [g for opt in self._inner() for g in opt.param_groups]
