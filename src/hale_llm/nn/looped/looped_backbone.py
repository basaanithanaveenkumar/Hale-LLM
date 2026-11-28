"""A decoder-only transformer whose middle layers are looped (Parcae, arXiv 2604.12946)."""

from __future__ import annotations

import math

import torch
from loguru import logger
from torch import nn

from hale_llm.nn.attention import cross_document_mask
from hale_llm.nn.blocks import build_blocks
from hale_llm.nn.blocks.residual_init import init_linear_and_embedding, scale_residual_outputs
from hale_llm.nn.looped.recurrence_sampler import RecurrenceSampler
from hale_llm.nn.looped.stable_injection import StableInjection
from hale_llm.nn.norm import build_norm
from hale_llm.nn.positional import build_positional_encoding


def split_layers(model_cfg) -> tuple[int, int, int]:
    """``(prelude, recurrent, coda)`` block counts: the config's, or thirds of ``n_layers`` (the paper loops "the middle third")."""
    prelude, recurrent, coda = model_cfg.loop_prelude_layers, model_cfg.loop_recurrent_layers, model_cfg.loop_coda_layers
    if prelude is None and recurrent is None and coda is None:
        edge = max(1, model_cfg.n_layers // 3)
        prelude, recurrent, coda = edge, model_cfg.n_layers - 2 * edge, edge
    if None in (prelude, recurrent, coda):
        raise ValueError("set all of loop_prelude_layers, loop_recurrent_layers and loop_coda_layers, or none of them")
    if min(prelude, recurrent, coda) < 1:
        raise ValueError(f"every unit needs at least one block, got prelude={prelude} recurrent={recurrent} coda={coda}")
    if prelude + recurrent + coda != model_cfg.n_layers:
        raise ValueError(
            f"prelude + recurrent + coda = {prelude + recurrent + coda} blocks must equal model.n_layers={model_cfg.n_layers}, "
            "so that the looped model has the same number of distinct blocks as the plain one"
        )
    return prelude, recurrent, coda


class LoopedBackbone(nn.Module):
    """``token ids -> hidden states -> logits`` with a prelude, a looped recurrent unit and a coda.

    Layout (Parcae Sec. 2.1 and 4.1, Algorithm 1)::

        tokens -> embedding -> P (prelude blocks) -> LN -> e                      e is injected at every loop
        h_0 ~ N(0, sigma^2)                                                       random initial state
        h_{t+1} = R( A_bar h_t + B_bar e )     for t = 0 .. T-1                   R = the recurrent blocks, with their residuals
        h_T -> readout matrix -> C (coda blocks) -> final norm -> lm_head -> logits

    ``R( A_bar h + B_bar e )`` is exactly ``A_bar h + B_bar e + R_bar(h, e)`` of the paper: the blocks' residual
    connections return their input (``A_bar h + B_bar e``) and add ``R_bar`` on top.

    Training draws a depth ``T`` for every sequence and backpropagates through the last ``mu_bwd`` loops only
    (Algorithm 2, :meth:`_run_loops`); evaluation uses a fixed number of loops (``loop_eval_recurrence``, default
    ``loop_mean_recurrence``) and a fixed initial state, so a validation loss does not move between calls.
    """

    def __init__(self, vocab_size: int, model_cfg) -> None:
        super().__init__()
        cfg = model_cfg
        self.vocab_size = vocab_size
        self.max_length = cfg.max_length
        self.n_prelude, self.n_recurrent, self.n_coda = split_layers(cfg)

        # 1. Positions, embedding and head: the same as the plain backbone.
        self.positional = build_positional_encoding(
            cfg.pos_encoding,
            d_model=cfg.d_model,
            max_length=cfg.max_length,
            head_dim=cfg.d_model // cfg.n_heads,
            rope_theta=cfg.rope_theta,
            rope_scaling=cfg.rope_scaling,
            rope_scaling_factor=cfg.rope_scaling_factor,
            rope_original_length=cfg.rope_original_length,
        )
        self.token_emb = nn.Embedding(vocab_size, cfg.d_model)
        # 2. The three units, all built from the same block factory (so ffn_type, norms and RoPE work unchanged).
        self.prelude = build_blocks(cfg, self.n_prelude, self.positional)
        self.prelude_norm = build_norm(cfg.norm_type, cfg.d_model)  # e = LN(P(s)): normalises the injected input (Sec. 3, J)
        self.injection = StableInjection(cfg.d_model, cfg.loop_decay_init)
        self.recurrent = build_blocks(cfg, self.n_recurrent, self.positional)
        self.readout = nn.Parameter(torch.eye(cfg.d_model))  # the paper's C matrix: recurrent width -> coda width
        self.coda = build_blocks(cfg, self.n_coda, self.positional)
        self.norm = build_norm(cfg.norm_type, cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, vocab_size, bias=False)

        # 3. How deep to loop.
        self.sampler = RecurrenceSampler(cfg.loop_mean_recurrence, cfg.loop_mean_backprop, cfg.loop_sampling)
        self.eval_recurrence = cfg.loop_eval_recurrence or cfg.loop_mean_recurrence
        # 4. The random initial state: std "like the weight init" (Parcae Table 17/18, "like-init"), N(0, 2 / (5 d)).
        self.state_std = cfg.loop_state_std if cfg.loop_state_std is not None else math.sqrt(2.0 / (5.0 * cfg.d_model))
        # Evaluation needs the SAME state for every call and for every prompt length (position t must not depend on the
        # length of the text, or the logits of a prefix would change when tokens are appended), so it is one fixed
        # [max_length, d] table. Built on the CPU explicitly: the dry run builds models under the "meta" device.
        generator = torch.Generator(device="cpu").manual_seed(cfg.loop_state_seed)
        self.register_buffer(
            "eval_state", torch.randn(cfg.max_length, cfg.d_model, generator=generator, device="cpu"), persistent=False
        )

        self._init_weights(cfg)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.token_emb.weight
        logger.info(
            "LoopedBackbone: prelude={} recurrent={} (x mu_rec={}, mu_bwd={}, {}) coda={} d_model={} A_bar0={}",
            self.n_prelude, self.n_recurrent, self.sampler.mean_recurrence, self.sampler.mean_backprop,
            self.sampler.mode, self.n_coda, cfg.d_model, cfg.loop_decay_init,
        )

    @property
    def layers(self) -> list[nn.Module]:
        """Every distinct transformer block once (the logger and the dry run's parameter count read this)."""
        return [*self.prelude, *self.recurrent, *self.coda]

    def _init_weights(self, cfg) -> None:
        """GPT-2 style, with the residual outputs scaled by the EFFECTIVE depth.

        Linears and embeddings start at N(0, 0.02). The recurrent blocks run ``mu_rec`` times, so the residual stream sees
        ``prelude + recurrent * mu_rec + coda`` blocks; the residual writers are scaled by ``1 / sqrt(2 * that)`` so the
        stream does not grow with the number of loops. (The injection parameters are plain Parameters and are not touched.)
        """
        init_linear_and_embedding(self)
        effective_depth = self.n_prelude + self.n_recurrent * self.sampler.mean_recurrence + self.n_coda
        residual_std = 0.02 / math.sqrt(2 * effective_depth)
        for block in self.layers:
            scale_residual_outputs(block, residual_std)

    # ----- the forward pass -----
    def encode(
        self,
        token_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        document_ids: torch.Tensor | None = None,
        num_loops: int | None = None,
    ) -> torch.Tensor:
        """``token_ids`` ``[B, S]`` to the final hidden states ``[B, S, D]``.

        ``num_loops`` forces every sequence to loop exactly that many times (also in training mode); without it, training
        samples a depth per sequence and evaluation uses ``loop_eval_recurrence``.
        """
        B, S = token_ids.shape
        if S > self.max_length:
            raise ValueError(f"sequence of {S} tokens exceeds max_length={self.max_length}")
        if num_loops is not None and num_loops < 1:
            raise ValueError(f"num_loops={num_loops} must be at least 1")

        # 1. Tokens to vectors, and the masks every block needs.
        hidden = self.positional.encode_embeddings(self.token_emb(token_ids))  # [B, S, D]
        key_padding = (attention_mask == 0) if attention_mask is not None else None  # [B, S]
        positions = torch.arange(S, device=token_ids.device)  # [S]
        document_mask = cross_document_mask(document_ids) if document_ids is not None else None  # [B, S, S]
        mask_args = (positions, key_padding, document_mask)

        # 2. Prelude P, then the normalised injection input e = LN(P(s)).
        for block in self.prelude:
            hidden = block(hidden, *mask_args)
        prelude_output = self.prelude_norm(hidden)  # e: [B, S, D]

        # 3. How many loops each sequence makes, and the random initial state h_0.
        depths = self._loop_depths(B, token_ids.device, num_loops)  # [B]
        state = self._initial_state(B, S, prelude_output)  # [B, S, D]

        # 4. The loops (Algorithm 2), then the readout matrix and the coda C.
        state = self._run_loops(state, prelude_output, depths, mask_args)  # [B, S, D]
        hidden = state @ self.readout.t()  # [B, S, D]
        for block in self.coda:
            hidden = block(hidden, *mask_args)
        return self.norm(hidden)  # [B, S, D]

    def forward(
        self,
        token_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        document_ids: torch.Tensor | None = None,
        num_loops: int | None = None,
    ) -> torch.Tensor:
        """``token_ids`` ``[B, S]`` to logits ``[B, S, vocab]``."""
        return self.lm_head(self.encode(token_ids, attention_mask, document_ids, num_loops))

    def _loop_depths(self, batch_size: int, device, num_loops: int | None) -> torch.Tensor:
        """``T`` for each sequence ``[B]``: forced, drawn per sequence (training), or fixed (evaluation)."""
        if num_loops is not None:
            return torch.full((batch_size,), num_loops, dtype=torch.long, device=device)
        if self.training:
            return self.sampler.sample(batch_size, device)
        return torch.full((batch_size,), self.eval_recurrence, dtype=torch.long, device=device)

    def _initial_state(self, batch_size: int, length: int, like: torch.Tensor) -> torch.Tensor:
        """``h_0 ~ N(0, sigma^2)`` ``[B, S, D]``: fresh noise in training, one fixed table in evaluation."""
        if self.training:
            noise = torch.randn(batch_size, length, like.size(-1), device=like.device)
        else:
            noise = self.eval_state[:length].to(like.device)[None].expand(batch_size, -1, -1)
        return (noise * self.state_std).to(like.dtype)

    def _recurrent_step(self, state: torch.Tensor, prelude_output: torch.Tensor, mask_args) -> torch.Tensor:
        """One loop: ``h' = R(A_bar h + B_bar e)``. The blocks' residuals make this ``A_bar h + B_bar e + R_bar``."""
        hidden = self.injection(state, prelude_output)  # [B, S, D]
        for block in self.recurrent:
            hidden = block(hidden, *mask_args)
        return hidden

    def _run_loops(self, state, prelude_output, depths: torch.Tensor, mask_args) -> torch.Tensor:
        """Algorithm 2 of the paper: per-sequence depths in one batch, gradient through the last ``mu_bwd`` loops only.

        Every sequence ``i`` has its own depth ``T_i``. All run for ``T_max = max(T_i)`` rounds, but a sequence with
        ``T_i < T_max`` first sits out ``tau_i = T_max - T_i`` rounds (its state is left at ``h_0``), so that all of them
        finish together and the last rounds, the ones with a gradient, are the last loops of every sequence.

            round t = 0 .. T_max - 1
              t <  tau_i                  no update          (the sequence has not started)
              tau_i <= t < T_max - mu_bwd  update, NO gradient
              t >= T_max - mu_bwd          update, WITH gradient
        """
        t_max = int(depths.max())
        wait = t_max - depths  # tau_i, [B]
        # Truncated backprop applies while training; evaluation keeps whatever the caller's grad mode says.
        grad_from = t_max - self.sampler.mean_backprop if self.training else 0
        some_sit_out = bool((wait > 0).any())  # False when every sequence has the same depth: no masking needed
        for t in range(t_max):
            if t < grad_from:
                with torch.no_grad():  # the early loops only move the state forward
                    new_state = self._recurrent_step(state, prelude_output, mask_args)
            else:
                new_state = self._recurrent_step(state, prelude_output, mask_args)
            if some_sit_out:
                started = (t >= wait)[:, None, None]  # [B, 1, 1]
                new_state = torch.where(started, new_state, state)  # sequences that have not started keep h_0
            state = new_state
        return state
