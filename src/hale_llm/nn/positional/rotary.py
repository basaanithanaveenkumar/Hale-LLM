"""Rotary position embedding (RoPE, Su et al. 2021)."""

from __future__ import annotations

import torch

from hale_llm.nn.positional.positional_encoding import PositionalEncoding
from hale_llm.nn.positional.registry import register_positional_encoding


@register_positional_encoding("rope")
class RotaryPositionalEncoding(PositionalEncoding):
    """Rotates pairs of query/key channels by an angle proportional to the position.

    Tiny example with head_dim=2: a token at position ``p`` has its (x0, x1) pair
    rotated by ``p * freq`` radians. The dot product of a query at position ``i`` and
    a key at position ``j`` then depends only on ``i - j``, so the model sees
    *relative* distance without any learned table, and it adds no parameters.

    With head_dim channels, channel pair ``m`` uses frequency ``theta ** (-m / half)``.
    Pairs follow the "split halves" convention: channel ``m`` pairs with ``m + half``.
    """

    name = "rope"

    def __init__(self, head_dim: int, rope_theta: float = 10_000.0, **_unused) -> None:
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError(f"RoPE needs an even head_dim, got {head_dim}")
        self.head_dim = head_dim
        self.rope_theta = rope_theta
        half = head_dim // 2
        # Frequencies never change, so keep them as a non-trainable buffer.
        inv_freq = 1.0 / (rope_theta ** (torch.arange(half, dtype=torch.float32) / half))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        first, second = x.chunk(2, dim=-1)
        return torch.cat((-second, first), dim=-1)

    def encode_queries_keys(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # angle[s, m] = position[s] * freq[m]; computed in float32 for accuracy.
        angles = torch.outer(positions.to(self.inv_freq.dtype), self.inv_freq)
        angles = torch.cat((angles, angles), dim=-1)  # (seq, head_dim)
        cos = angles.cos()[None, None].to(q.dtype)
        sin = angles.sin()[None, None].to(q.dtype)
        # Rotation: x * cos + rotate_half(x) * sin
        return (
            q * cos + self._rotate_half(q) * sin,
            k * cos + self._rotate_half(k) * sin,
        )

    def describe(self) -> str:
        return f"rope(theta={self.rope_theta:g}, head_dim={self.head_dim})"
