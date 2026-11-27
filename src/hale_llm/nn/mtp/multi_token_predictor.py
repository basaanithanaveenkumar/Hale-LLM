"""The stack of multi-token-prediction depths that sits on top of the trunk."""

from __future__ import annotations

import math

import torch
from torch import nn

from hale_llm.nn.attention import cross_document_mask
from hale_llm.nn.blocks.residual_init import init_linear_and_embedding, scale_residual_outputs
from hale_llm.nn.mtp.mtp_module import MTPModule
from hale_llm.nn.positional.positional_encoding import PositionalEncoding


class MultiTokenPredictor(nn.Module):
    """``model.mtp_depth`` chained :class:`MTPModule` s. Depth ``k`` predicts the token ``k + 1`` steps ahead.

    Depth 1 reads the trunk's final hidden states, depth ``k`` reads depth ``k - 1``'s output, so the
    causal chain from the DeepSeek-V3 paper is kept. ``token_emb`` and ``lm_head`` are the trunk's own
    and are passed in rather than stored, so their weights exist once.
    """

    offset_conditioned = False  # see OffsetConditionedPredictor
    chained = True  # each depth reads the previous depth's output, not the trunk's

    def __init__(self, model_cfg, positional: PositionalEncoding | None) -> None:
        super().__init__()
        self.depth = model_cfg.mtp_depth
        if model_cfg.max_length < self.depth + 2:
            raise ValueError(
                f"mtp_depth={self.depth} needs a context window of at least {self.depth + 2} tokens, "
                f"got {model_cfg.max_length}"
            )
        block_cfg = model_cfg.mtp_block_config()  # the trunk's settings, plus any mtp_* overrides
        self.depths = nn.ModuleList(
            MTPModule(block_cfg, model_cfg.mtp_n_layers, positional) for _ in range(self.depth)
        )
        # Same initialisation as the trunk's blocks.
        init_linear_and_embedding(self)
        residual_std = 0.02 / math.sqrt(2 * model_cfg.n_layers)
        for module in self.depths:
            for block in module.blocks:
                scale_residual_outputs(block, residual_std)

    def forward(
        self,
        trunk_hidden: torch.Tensor,
        token_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
        token_emb: nn.Module,
        lm_head: nn.Module,
        document_ids: torch.Tensor | None = None,
    ) -> list[torch.Tensor]:
        """Logits of every depth, for training.

        ``trunk_hidden`` is ``[B, T, D]`` and ``token_ids`` ``[B, T]``. Depth ``k`` (1-based) returns
        ``[B, T - k, V]``: its position ``i`` is the prediction of ``t_{i+k+1}`` and uses tokens up to ``t_{i+k}``.
        The last position has no target, so the loss drops it.
        """
        B, T = token_ids.shape
        if T < self.depth + 2:
            raise ValueError(
                f"mtp_depth={self.depth} needs sequences of at least {self.depth + 2} tokens, got {T}"
            )
        previous = trunk_hidden  # [B, T, D]
        document_mask = cross_document_mask(document_ids) if document_ids is not None else None  # [B, T, T]
        all_logits = []
        for k, module in enumerate(self.depths, start=1):
            length = T - k  # the positions that still have a token k steps ahead
            # 1. Embed t_{i+k} for i = 0 .. T-k-1.
            embedded = token_emb(token_ids[:, k:])  # [B, T-k, D]
            # 2. Previous depth's hidden state at the same positions (drop its last one: no t_{i+k} for it).
            hidden_in = previous[:, :length]  # [B, T-k, D]
            # 3. Padding and positions of this depth's (shorter) sequence.
            key_padding = (attention_mask[:, :length] == 0) if attention_mask is not None else None
            positions = torch.arange(length, device=token_ids.device)
            # 4. One MTP depth, then the shared head.
            depth_document_mask = document_mask[:, :length, :length] if document_mask is not None else None
            hidden = module(hidden_in, embedded, positions, key_padding, depth_document_mask)  # [B, T-k, D]
            all_logits.append(module.logits(hidden, lm_head))  # [B, T-k, V]
            previous = hidden
        return all_logits

    def step(
        self,
        k: int,
        previous_hidden: torch.Tensor,
        tokens: torch.Tensor,
        token_emb: nn.Module,
        lm_head: nn.Module,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run depth ``k`` (1-based) over a known prefix, for speculative decoding.

        ``previous_hidden`` is ``[B, L, D]``; ``tokens`` is ``[B, >= L + k]`` and must hold ``t_0 .. t_{L+k-1}``.
        Returns this depth's hidden states ``[B, L, D]`` (the next depth reads them) and the vocabulary
        scores of its **last** position, ``[B, V]``, a prediction of ``t_{L+k}``. It is the same computation
        as :meth:`forward`, so training and inference cannot drift apart.
        """
        L = previous_hidden.size(1)
        module = self.depths[k - 1]
        embedded = token_emb(tokens[:, k : k + L])  # t_{i+k} for i = 0 .. L-1, [B, L, D]
        positions = torch.arange(L, device=tokens.device)
        hidden = module(previous_hidden, embedded, positions)  # [B, L, D]
        return hidden, module.logits(hidden[:, -1], lm_head)  # [B, L, D], [B, V]
