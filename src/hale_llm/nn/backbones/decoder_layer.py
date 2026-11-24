"""One pre-norm transformer block with a configurable norm and position scheme."""

from __future__ import annotations

import torch
from torch import nn

from hale_llm.nn.norm import build_norm


class DecoderLayer(nn.Module):
    """``x + attention(norm(x))`` followed by ``x + ffn(norm(x))``.

    When ``use_time_cond`` is on (diffusion variants), a zero-initialised linear
    layer turns the time embedding into a scale and shift for each norm output,
    so training starts from the unconditioned model.
    """

    def __init__(
        self,
        d_model: int,
        attention: nn.Module,
        ffn: nn.Module,
        norm_type: str,
        use_time_cond: bool = False,
    ) -> None:
        super().__init__()
        self.attn = attention
        self.ff = ffn
        self.ln1 = build_norm(norm_type, d_model)
        self.ln2 = build_norm(norm_type, d_model)
        self.use_time_cond = use_time_cond
        if use_time_cond:
            self.time_proj = nn.Linear(d_model, 4 * d_model)
            nn.init.zeros_(self.time_proj.weight)
            nn.init.zeros_(self.time_proj.bias)

    @staticmethod
    def _modulate(h: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor) -> torch.Tensor:
        # A per-sequence time embedding is (batch, d); broadcast it over tokens.
        if scale.dim() == 2:
            scale, shift = scale.unsqueeze(1), shift.unsqueeze(1)
        return h * (1 + scale) + shift

    def forward(
        self,
        x: torch.Tensor,
        t_emb: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
        key_padding_mask: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        scale1 = shift1 = scale2 = shift2 = None
        if self.use_time_cond and t_emb is not None:
            scale1, shift1, scale2, shift2 = self.time_proj(t_emb).chunk(4, dim=-1)

        # Attention sub-block.
        h = self.ln1(x)
        if scale1 is not None:
            h = self._modulate(h, scale1, shift1)
        x = x + self.attn(
            h, attn_mask=attn_mask, key_padding_mask=key_padding_mask, positions=positions
        )

        # Feed-forward sub-block.
        h = self.ln2(x)
        if scale2 is not None:
            h = self._modulate(h, scale2, shift2)
        return x + self.ff(h)
