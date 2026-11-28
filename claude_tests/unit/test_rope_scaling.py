"""RoPE frequency scaling (none / linear / YaRN) and changing theta after construction."""

import math

import pytest
import torch

from hale_llm.nn.positional import RotaryPositionalEncoding
from hale_llm.nn.positional.scaling import ROPE_SCALINGS, get_rope_scaling

HEAD_DIM, THETA, ORIGINAL = 64, 10_000.0, 4096


def test_strategies_are_registered():
    assert {"none", "linear", "yarn"} <= set(ROPE_SCALINGS)
    with pytest.raises(KeyError, match="registered"):
        get_rope_scaling("magic")


def test_none_is_plain_rope():
    inv, scale = get_rope_scaling("none").compute(HEAD_DIM, THETA, 1.0, ORIGINAL)
    assert scale == 1.0
    assert inv[0].item() == pytest.approx(1.0)  # first channel pair rotates once per position
    assert inv[-1].item() == pytest.approx(THETA ** (-(HEAD_DIM // 2 - 1) / (HEAD_DIM // 2)))


def test_linear_divides_every_frequency():
    base, _ = get_rope_scaling("none").compute(HEAD_DIM, THETA, 1.0, ORIGINAL)
    inv, scale = get_rope_scaling("linear").compute(HEAD_DIM, THETA, 4.0, ORIGINAL)
    assert torch.allclose(inv, base / 4) and scale == 1.0


def test_yarn_with_factor_one_changes_nothing():
    base, _ = get_rope_scaling("none").compute(HEAD_DIM, THETA, 1.0, ORIGINAL)
    inv, scale = get_rope_scaling("yarn").compute(HEAD_DIM, THETA, 1.0, ORIGINAL)
    assert torch.equal(inv, base) and scale == 1.0


def test_yarn_keeps_fast_channels_and_interpolates_slow_ones():
    factor = 8.0
    base, _ = get_rope_scaling("none").compute(HEAD_DIM, THETA, 1.0, ORIGINAL)
    inv, scale = get_rope_scaling("yarn").compute(HEAD_DIM, THETA, factor, ORIGINAL)
    assert inv[0].item() == pytest.approx(base[0].item())  # fastest channel untouched
    assert inv[-1].item() == pytest.approx(base[-1].item() / factor)  # slowest fully interpolated
    assert scale == pytest.approx(0.1 * math.log(factor) + 1.0)
    # in between: never faster than the original, never slower than fully interpolated
    assert torch.all(inv <= base + 1e-9) and torch.all(inv >= base / factor - 1e-9)


def test_yarn_rejects_a_context_too_short_for_its_boundaries():
    with pytest.raises(ValueError, match="original_length"):
        get_rope_scaling("yarn").compute(HEAD_DIM, THETA, 4.0, 32)


def test_set_theta_changes_the_rotation_and_validates():
    rope = RotaryPositionalEncoding(head_dim=8, rope_theta=10_000.0)
    q = k = torch.randn(1, 1, 5, 8)
    before, _ = rope.encode_queries_keys(q, k, torch.arange(5))
    rope.set_theta(1_000_000.0)
    after, _ = rope.encode_queries_keys(q, k, torch.arange(5))
    assert not torch.allclose(before, after)
    assert "1e+06" in rope.describe()
    with pytest.raises(ValueError):
        rope.set_theta(1.0)


def test_yarn_attention_scale_is_applied_to_queries_and_keys():
    rope = RotaryPositionalEncoding(
        head_dim=HEAD_DIM, rope_theta=THETA, rope_scaling="yarn", rope_scaling_factor=4.0, rope_original_length=ORIGINAL
    )
    q = torch.ones(1, 1, 1, HEAD_DIM)
    rq, _ = rope.encode_queries_keys(q, q, torch.tensor([0]))  # position 0: no rotation, only scaling
    assert rq[0, 0, 0, 0].item() == pytest.approx(0.1 * math.log(4.0) + 1.0, rel=1e-5)
    assert "yarn" in rope.describe()
