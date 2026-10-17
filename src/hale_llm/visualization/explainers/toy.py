"""Train a tiny model of any registered variant on one sentence, for an explainer animation.

The loop is the trainer's inner step (registered loss + architecture auxiliary losses, AdamW,
gradient clipping) without checkpoints, logging or data downloads, so an explainer can build
its model in seconds on a CPU and the animation shows *trained* behaviour, not noise.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import torch

from hale_llm.config.run import RunConfig
from hale_llm.core.registry import get_loss, get_variant
from hale_llm.dataloader.collate import get_collate
from hale_llm.exceptions import VisualizationError
from hale_llm.losses import with_auxiliary_losses
from hale_llm.visualization.explainers.tokenizer import CharTokenizer


@dataclass
class ToyRun:
    """A trained toy model and what produced it (``text`` = the first training sentence)."""

    variant: str
    model: torch.nn.Module
    tokenizer: CharTokenizer
    cfg: RunConfig
    texts: list[str]
    losses: list[float] = field(default_factory=list)

    @property
    def text(self) -> str:
        """The first training sentence."""
        return self.texts[0]

    @property
    def ids(self) -> torch.Tensor:
        """``(1, len(text))`` ids of the first training sentence."""
        return torch.tensor([self.tokenizer.encode(self.text)])


def toy_config(variant: str, text: str, model: dict[str, Any] | None = None,
               sample: dict[str, Any] | None = None) -> RunConfig:
    """Run config of a toy model whose ``max_length`` is the sentence length."""
    model_cfg = {"d_model": 64, "n_heads": 4, "n_layers": 2, "d_ff": 128, "dropout": 0.0,
                 "max_length": len(text), **(model or {})}
    return RunConfig.model_validate({
        "variant": variant,
        "model": model_cfg,
        "sample": {"eps": 1e-3, **(sample or {})},
        "device": "cpu",
    })


def train_toy(
    variant: str,
    texts: str | Sequence[str],
    *,
    model: dict[str, Any] | None = None,
    sample: dict[str, Any] | None = None,
    steps: int = 400,
    lr: float = 3e-3,
    batch_size: int = 16,
    seed: int = 0,
) -> ToyRun:
    """Fit a ``variant`` model to ``texts`` (equal-length sentences, cycled to ``batch_size``).

    Several sentences that share most characters give the model real choice points, so the
    animations show genuine uncertainty where the sentences branch.
    """
    import hale_llm.models  # noqa: F401  - registers the variants

    texts = [texts] if isinstance(texts, str) else list(texts)
    if len({len(t) for t in texts}) != 1:
        raise VisualizationError("toy sentences must have equal length (one fixed canvas)")
    torch.manual_seed(seed)
    tokenizer = CharTokenizer()
    cfg = toy_config(variant, texts[0], model, sample)
    net = get_variant(variant)(vocab_size=tokenizer.vocab_size, cfg=cfg)
    loss_fn = get_loss(variant)
    rows = [tokenizer.encode(texts[i % len(texts)]) for i in range(max(batch_size, len(texts)))]
    ids = torch.tensor(rows)
    batch = get_collate(variant)(ids, tokenizer, cfg=cfg)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=lr * 0.05)
    losses: list[float] = []
    net.train()
    for _ in range(steps):
        loss, _extras = with_auxiliary_losses(net, loss_fn(net, batch))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        sched.step()
        losses.append(float(loss.detach()))
    net.eval()
    return ToyRun(variant, net, tokenizer, cfg, texts, losses)
