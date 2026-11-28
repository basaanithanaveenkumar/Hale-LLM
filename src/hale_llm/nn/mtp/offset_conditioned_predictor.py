"""One multi-token-prediction module that predicts at a randomly chosen offset (offset-conditioned MTP)."""

from __future__ import annotations

import math

import torch
from torch import nn

from hale_llm.nn.attention import cross_document_mask
from hale_llm.nn.blocks.residual_init import init_linear_and_embedding, scale_residual_outputs
from hale_llm.nn.mtp.offset_share import OffsetShare
from hale_llm.nn.mtp.mtp_module import MTPModule
from hale_llm.nn.mtp.stratified_offset_assigner import StratifiedOffsetAssigner
from hale_llm.nn.positional.positional_encoding import PositionalEncoding

IGNORE_INDEX = -100  # the label value the loss skips (same as everywhere else in the code base)


class OffsetConditionedPredictor(nn.Module):
    """DeepSeek-V3's MTP module, but with the offset ``n`` as an input instead of a fixed depth.

    DeepSeek-V3 stacks ``D`` modules; module ``k`` always predicts ``k`` tokens beyond the main head. Here there is ONE
    module (``mtp_module``, the same :class:`MTPModule`), and every training step each sequence gets an offset
    ``n`` in ``1 .. mtp_depth``, assigned stratified over the whole step (see :class:`StratifiedOffsetAssigner`).
    For a position ``t``::

        main head        h(t)                                  ->  x(t+1)      unchanged
        this module      h(t)  +  x(t+n)  +  offset-embedding(n)  ->  x(t+n+1)

    The input token is the real ``x(t+n)`` and the target is the token after it, ``x(t+n+1)``. Example on
    "the cat sat on the mat" with ``t`` at "the": ``n=1`` reads "cat" and predicts "sat" (exactly DeepSeek's depth 1),
    ``n=2`` reads "sat" and predicts "on", ``n=3`` reads "on" and predicts "the". The module never reads ``x(t+1)``
    at position ``t`` when ``n=2``; it only sees it through the previous position's input (``x(t-1+2)``).

    The learned offset embedding is added to the token embedding, so the module can tell the offsets apart. Unlike the
    chained depths of :class:`MultiTokenPredictor`, every offset reads the trunk's hidden state directly (that is the
    "same trunk state" the drafts of speculative decoding reuse), so ``chained`` is False. ``token_emb`` and
    ``lm_head`` are the trunk's own and are passed in, so they exist once.
    """

    offset_conditioned = True
    chained = False

    def __init__(
        self, model_cfg, positional: PositionalEncoding | None, seed: int = 0, accum_steps: int = 1
    ) -> None:
        super().__init__()
        self.depth = model_cfg.mtp_depth  # the largest offset; named like MultiTokenPredictor's so shared code reads it
        if model_cfg.max_length < self.depth + 2:
            raise ValueError(
                f"mtp_depth={self.depth} needs a context window of at least {self.depth + 2} tokens, "
                f"got {model_cfg.max_length}"
            )
        block_cfg = model_cfg.mtp_block_config()  # the trunk's settings, plus any mtp_* overrides
        self.mtp_module = MTPModule(block_cfg, model_cfg.mtp_n_layers, positional)
        self.offset_emb = nn.Embedding(self.depth, model_cfg.d_model)  # row n-1 is the embedding of offset n
        self.assigner = StratifiedOffsetAssigner(
            self.depth, seed, accum_steps, model_cfg.mtp_offset_remainder,
            model_cfg.mtp_offsets_per_sequence, model_cfg.mtp_offset_stratified,
        )
        # Same initialisation as the trunk's blocks.
        init_linear_and_embedding(self)
        residual_std = 0.02 / math.sqrt(2 * model_cfg.n_layers)
        for block in self.mtp_module.blocks:
            scale_residual_outputs(block, residual_std)

    # ----- the offsets -----
    def next_offsets(self, batch_size: int, device: torch.device | str) -> OffsetShare:
        """The offsets of the next micro-batch of ``batch_size`` sequences (stratified over the whole step)."""
        return self.assigner.next_share(batch_size, device)

    @staticmethod
    def _ahead(values: torch.Tensor, offsets: torch.Tensor, extra: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
        """``values[b, t + offsets[b] + extra]`` for every ``(b, t)``, and whether that index exists.

        Returns ``(gathered [B, T], exists [B, T])``. Where the index runs past the end the gathered value is a
        clamped placeholder and ``exists`` is False; callers must ignore those positions.
        """
        B, T = values.shape
        index = torch.arange(T, device=values.device)[None, :] + offsets[:, None] + extra  # [B, T]
        exists = index < T  # [B, T]
        return values.gather(1, index.clamp(max=T - 1)), exists

    # ----- training -----
    def forward(
        self,
        trunk_hidden: torch.Tensor,
        token_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
        token_emb: nn.Module,
        lm_head: nn.Module,
        offsets: torch.Tensor,
        document_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Logits ``[B, T, V]``: position ``t`` of sequence ``b`` predicts ``x(t + n_b + 1)``, ``n_b = offsets[b]``.

        ``trunk_hidden`` is ``[B, T, D]``, ``token_ids`` ``[B, T]``, ``offsets`` ``[B]`` with values in
        ``1 .. mtp_depth``. Positions with no target (``t + n_b + 1 >= T``) still get logits, built from a placeholder
        input; :meth:`targets` marks them to be ignored, and the causal attention means they never feed a valid one.
        """
        B, T = token_ids.shape
        self._check_inputs(T, offsets, B)
        # 1. Algorithm step: for each sequence b, take the real token n_b steps ahead of every position.
        ahead_tokens, _ = self._ahead(token_ids, offsets)  # x(t + n_b), [B, T]
        # 2. Embed it and add the learned embedding of the offset (the same vector at every position of a sequence).
        embedded = token_emb(ahead_tokens) + self.offset_emb(offsets - 1)[:, None, :]  # [B, T, D]
        # 3. Padding, positions and (if used) the document mask, exactly as for the trunk.
        key_padding = (attention_mask == 0) if attention_mask is not None else None  # [B, T]
        positions = torch.arange(T, device=token_ids.device)  # [T]
        document_mask = cross_document_mask(document_ids) if document_ids is not None else None  # [B, T, T]
        # 4. One MTP module over the trunk's own hidden states (not a previous depth's), then the shared head.
        hidden = self.mtp_module(trunk_hidden, embedded, positions, key_padding, document_mask)  # [B, T, D]
        return self.mtp_module.logits(hidden, lm_head)  # [B, T, V]

    def targets(
        self, labels: torch.Tensor, offsets: torch.Tensor, document_ids: torch.Tensor | None = None
    ) -> torch.Tensor:
        """The target of every position of :meth:`forward`: ``x(t + n_b + 1)``, or ``-100`` where it must not count.

        ``labels[i]`` is the token after position ``i`` (``-100`` for padding and for the last position), so the
        target of position ``t`` is ``labels[t + n_b]``. A target is dropped (set to ``-100``) when

        * it lies past the end of the sequence (``t + n_b + 1 >= T``), or the label itself is ``-100`` (padding);
        * with ``document_ids``: it belongs to another document than position ``t`` (the token after an
          end-of-text token starts a new document, and the module must not be asked to guess it).
        """
        B, T = labels.shape
        # 1. Algorithm step: pick labels[t + n_b], the token after the module's input x(t + n_b).
        targets, label_exists = self._ahead(labels, offsets)  # [B, T], [B, T]
        # 2. The target token x(t + n_b + 1) must exist, i.e. the label index must be at most T - 2.
        target_exists = label_exists & (
            torch.arange(T, device=labels.device)[None, :] + offsets[:, None] <= T - 2
        )  # [B, T]
        keep = target_exists
        # 3. With several documents in the window, keep only targets from the same document as the position.
        if document_ids is not None:
            target_document, _ = self._ahead(document_ids, offsets, extra=1)  # document of x(t + n_b + 1)
            keep = keep & (target_document == document_ids)
        return torch.where(keep, targets, torch.full_like(targets, IGNORE_INDEX))

    def _check_inputs(self, T: int, offsets: torch.Tensor, B: int) -> None:
        if T < self.depth + 2:
            raise ValueError(f"mtp_depth={self.depth} needs sequences of at least {self.depth + 2} tokens, got {T}")
        if offsets.shape != (B,):
            raise ValueError(f"offsets must have shape [{B}] (one per sequence), got {tuple(offsets.shape)}")
        if offsets.numel() and (int(offsets.min()) < 1 or int(offsets.max()) > self.depth):
            raise ValueError(f"offsets must be between 1 and mtp_depth={self.depth}")

    # ----- inference -----
    def step(
        self,
        k: int,
        trunk_hidden: torch.Tensor,
        tokens: torch.Tensor,
        token_emb: nn.Module,
        lm_head: nn.Module,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Offset ``k`` over a known prefix, for speculative decoding; the same computation as :meth:`forward`.

        ``trunk_hidden`` is ``[B, L, D]`` and ``tokens`` is ``[B, >= L + k]`` holding ``x_0 .. x_{L+k-1}``. Returns the
        module's hidden states ``[B, L, D]`` and the vocabulary scores of its last position, ``[B, V]``, a prediction
        of ``x_{L+k}``. Every offset reads the trunk's states, so the caller passes the same ``trunk_hidden`` each time.
        """
        if not 1 <= k <= self.depth:
            raise ValueError(f"offset k={k} must be between 1 and mtp_depth={self.depth}")
        L = trunk_hidden.size(1)
        embedded = token_emb(tokens[:, k : k + L]) + self.offset_emb.weight[k - 1]  # x(i+k) + offset k, [B, L, D]
        positions = torch.arange(L, device=tokens.device)
        hidden = self.mtp_module(trunk_hidden, embedded, positions)  # [B, L, D]
        return hidden, self.mtp_module.logits(hidden[:, -1], lm_head)  # [B, L, D], [B, V]
