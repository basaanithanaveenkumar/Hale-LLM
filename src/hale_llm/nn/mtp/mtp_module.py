"""One multi-token-prediction depth (DeepSeek-V3 style)."""

from __future__ import annotations

import torch
from torch import nn

from hale_llm.nn.blocks import build_blocks
from hale_llm.nn.norm import RMSNorm
from hale_llm.nn.positional.positional_encoding import PositionalEncoding


class MTPModule(nn.Module):
    """Predicts the token ``k`` steps further ahead than the trunk, from the previous depth's hidden state.

    At position ``i`` it combines ``h`` (the hidden state of the previous depth at ``i``) with the
    embedding of token ``t_{i+k}``, runs one causal transformer block, and the shared output head
    reads the result as a prediction of ``t_{i+k+1}``::

        merged = proj( [ RMSNorm(h_i) ; RMSNorm(Emb(t_{i+k})) ] )      # [2D] -> [D]
        out    = blocks(merged)               # mtp_n_layers causal transformer blocks within this depth
        logits = lm_head( RMSNorm(out) )                                # shared head, own output norm

    The embedding table and the output head are not owned here: they are shared with the trunk and
    passed in by the caller. The output norm matters because the trunk's head only ever sees
    normalised features (the trunk ends in a final norm).
    """

    def __init__(self, block_cfg, n_layers: int, positional: PositionalEncoding | None) -> None:
        """``block_cfg`` is the model config the blocks are built from (see ``ModelConfig.mtp_block_config``)."""
        super().__init__()
        d_model = block_cfg.d_model
        self.hidden_norm = RMSNorm(d_model)  # on the previous depth's hidden state
        self.embed_norm = RMSNorm(d_model)  # on the embedding of the token k steps ahead
        self.proj = nn.Linear(2 * d_model, d_model, bias=False)  # [2D] -> [D]
        self.blocks = build_blocks(block_cfg, n_layers, positional)  # the same helper the trunk uses
        self.head_norm = RMSNorm(d_model)  # before the shared output head

    def forward(
        self,
        previous_hidden: torch.Tensor,
        embedded: torch.Tensor,
        positions: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """``previous_hidden`` and ``embedded`` are both ``[B, L, D]``; returns ``[B, L, D]``."""
        # 1. Normalise both inputs, put them side by side, and project back to the model width.
        merged = torch.cat(
            [self.hidden_norm(previous_hidden), self.embed_norm(embedded)], dim=-1
        )  # [B, L, 2D]
        merged = self.proj(merged)  # [B, L, D]
        # 2. The causal transformer block(s) over the L positions of this depth.
        for block in self.blocks:
            merged = block(merged, positions, key_padding_mask)  # [B, L, D]
        return merged

    def logits(self, hidden: torch.Tensor, lm_head: nn.Module) -> torch.Tensor:
        """Vocabulary scores from this depth's hidden state, using the trunk's output head: ``[B, L, V]``."""
        return lm_head(self.head_norm(hidden))
