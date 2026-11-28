"""Attention heads: scaling, causality, padding, positional hook and weights output."""

import math

import pytest
import torch

from hale_llm.nn.attention import HeadAttention, MultiHeadAttention
from hale_llm.nn.positional import RotaryPositionalEncoding


def _head(**kwargs):
    torch.manual_seed(0)
    return HeadAttention(d_model=16, head_size=8, causal=True, **kwargs).eval()


def test_output_shapes():
    x = torch.randn(2, 5, 16)
    assert _head()(x).shape == (2, 5, 8)
    assert MultiHeadAttention(16, 4, causal=True)(x).shape == (2, 5, 16)


def test_scores_are_scaled_by_head_size_not_model_width():
    # Weights returned by the explicit path must match a hand computation with 1/sqrt(head_size).
    head = _head(return_attn_weights=True)
    x = torch.randn(1, 4, 16)
    _, weights = head(x)
    scores = head.query_proj(x) @ head.key_proj(x).transpose(-2, -1) / math.sqrt(8)
    scores = scores.masked_fill(torch.triu(torch.ones(4, 4, dtype=torch.bool), 1), float("-inf"))
    assert torch.allclose(weights, scores.softmax(-1), atol=1e-6)


def test_fused_and_explicit_paths_agree():
    x = torch.randn(2, 6, 16)
    torch.manual_seed(1)
    fused = HeadAttention(16, 8, causal=True).eval()
    explicit = HeadAttention(16, 8, causal=True, return_attn_weights=True).eval()
    explicit.load_state_dict(fused.state_dict())
    out, weights = explicit(x)
    assert torch.allclose(fused(x), out, atol=1e-5)
    assert torch.allclose(weights.sum(-1), torch.ones(2, 6), atol=1e-5)


def test_causal_head_ignores_the_future():
    head = _head()
    a = torch.randn(1, 6, 16)
    b = a.clone()
    b[:, 4:] += 1.0
    assert torch.allclose(head(a)[:, :4], head(b)[:, :4], atol=1e-6)


def test_padded_keys_are_ignored_and_stay_finite():
    head = _head()
    x = torch.randn(1, 5, 16)
    pad = torch.tensor([[True, True, False, False, False]])  # left padding
    out = head(x, key_padding_mask=pad)
    assert torch.isfinite(out).all()
    changed = x.clone()
    changed[:, :2] += 5.0  # padded tokens must not influence real positions
    assert torch.allclose(out[:, 2:], head(changed, key_padding_mask=pad)[:, 2:], atol=1e-5)


def test_rope_is_applied_through_the_positional_hook():
    rope = RotaryPositionalEncoding(head_dim=8)
    torch.manual_seed(0)
    plain = HeadAttention(16, 8, causal=True).eval()
    rotated = HeadAttention(16, 8, causal=True, positional=rope).eval()
    rotated.load_state_dict(plain.state_dict())
    x = torch.randn(1, 6, 16)
    assert not torch.allclose(plain(x), rotated(x), atol=1e-4)


def test_heads_must_divide_model_width():
    with pytest.raises(ValueError, match="divisible"):
        MultiHeadAttention(10, 4)
