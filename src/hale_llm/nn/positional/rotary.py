"""Rotary position embedding (RoPE, Su et al. 2021)."""

from __future__ import annotations

import torch

from hale_llm.nn.positional.positional_encoding import PositionalEncoding
from hale_llm.nn.positional.registry import register_positional_encoding
from hale_llm.nn.positional.scaling import get_rope_scaling


@register_positional_encoding("rope")
class RotaryPositionalEncoding(PositionalEncoding):
    """Rotates pairs of query/key channels by an angle proportional to the position.

    Tiny example with head_dim=2: a token at position ``p`` has its (x0, x1) pair
    rotated by ``p * freq`` radians. The dot product of a query at position ``i`` and
    a key at position ``j`` then depends only on ``i - j``, so the model sees
    *relative* distance without any learned table, and it adds no parameters.

    With head_dim channels, channel pair ``m`` uses frequency ``theta ** (-m / half)``.
    Pairs follow the "split halves" convention: channel ``m`` pairs with ``m + half``.

    The base ``theta`` and a scaling strategy (``model.rope_scaling``) can be changed
    after construction, which is how context extension works: raise ``theta`` for a
    longer training context, or add YaRN scaling to run beyond the trained length.
    """

    name = "rope"

    def __init__(
        self,
        head_dim: int,
        rope_theta: float = 10_000.0,
        rope_scaling: str = "none",
        rope_scaling_factor: float = 1.0,
        rope_original_length: int = 4096,
        **_unused,
    ) -> None:
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError(f"RoPE needs an even head_dim, got {head_dim}")
        self.head_dim = head_dim
        self.rope_theta = rope_theta
        self._scaling_name = rope_scaling
        self._scaling_factor = rope_scaling_factor
        self._original_length = rope_original_length
        self._attention_scale = 1.0
        self.register_buffer("inv_freq", torch.empty(head_dim // 2), persistent=False)
        self._refresh()

    def _refresh(self) -> None:
        """Recompute the frequencies from the current theta and scaling."""
        inv_freq, self._attention_scale = get_rope_scaling(self._scaling_name).compute(
            self.head_dim, self.rope_theta, self._scaling_factor, self._original_length
        )
        self.inv_freq = inv_freq.to(self.inv_freq.device)

    def set_theta(self, theta: float) -> None:
        """Change the base frequency (used when the training context is extended)."""
        if theta <= 1:
            raise ValueError(f"rope theta must be > 1, got {theta}")
        self.rope_theta = theta
        self._refresh()

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
        cos = (angles.cos() * self._attention_scale)[None, None].to(q.dtype)
        sin = (angles.sin() * self._attention_scale)[None, None].to(q.dtype)
        # Rotation: x * cos + rotate_half(x) * sin
        return (
            q * cos + self._rotate_half(q) * sin,
            k * cos + self._rotate_half(k) * sin,
        )

    def describe(self) -> str:
        scaling = "" if self._scaling_name == "none" else f", {self._scaling_name} x{self._scaling_factor:g}"
        return f"rope(theta={self.rope_theta:g}, head_dim={self.head_dim}{scaling})"
