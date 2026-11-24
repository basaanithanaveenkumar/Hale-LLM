"""Positional encodings, norm layers and the backbone factory that wires them from config."""

import pytest
import torch

from hale_llm.config.sections.model import ModelConfig
from hale_llm.nn.backbones import GPTBackbone, build_backbone
from hale_llm.nn.norm import NORMS, RMSNorm, build_norm
from hale_llm.nn.positional import (
    POSITIONAL_ENCODINGS,
    LearnedPositionalEncoding,
    RotaryPositionalEncoding,
    build_positional_encoding,
)


def test_both_schemes_and_norms_are_registered():
    assert {"rope", "learned"} <= set(POSITIONAL_ENCODINGS)
    assert {"layer", "rms"} <= set(NORMS)


def test_rope_keeps_vector_length():
    # A rotation never changes a vector's norm.
    rope = RotaryPositionalEncoding(head_dim=8)
    q = torch.randn(1, 2, 5, 8)
    rq, _ = rope.encode_queries_keys(q, q.clone(), torch.arange(5))
    assert torch.allclose(rq.norm(dim=-1), q.norm(dim=-1), atol=1e-5)


def test_rope_scores_depend_only_on_relative_distance():
    rope = RotaryPositionalEncoding(head_dim=8)
    q_vec, k_vec = torch.randn(8), torch.randn(8)

    def score(pos_q: int, pos_k: int) -> float:
        # Place the same vectors at the given positions and take their dot product.
        q = q_vec.expand(1, 1, 1, 8).contiguous()
        k = k_vec.expand(1, 1, 1, 8).contiguous()
        rq, _ = rope.encode_queries_keys(q, q, torch.tensor([pos_q]))
        _, rk = rope.encode_queries_keys(k, k, torch.tensor([pos_k]))
        return (rq * rk).sum().item()

    assert score(3, 1) == pytest.approx(score(10, 8), abs=1e-4)  # distance 2 both times
    assert score(3, 1) != pytest.approx(score(3, 0), abs=1e-4)  # distance 2 vs 3


def test_rope_rejects_odd_head_dim():
    with pytest.raises(ValueError):
        RotaryPositionalEncoding(head_dim=7)


def test_learned_adds_table_and_handles_doubled_sequence():
    enc = LearnedPositionalEncoding(d_model=4, max_length=6)
    assert enc.encode_embeddings(torch.zeros(2, 6, 4)).shape == (2, 6, 4)
    doubled = enc.encode_embeddings(torch.zeros(1, 12, 4))  # clean + noised halves
    assert torch.equal(doubled[:, :6], doubled[:, 6:])
    with pytest.raises(ValueError):
        enc.encode_embeddings(torch.zeros(1, 14, 4))


def test_unknown_names_raise_with_options():
    with pytest.raises(KeyError, match="registered"):
        build_positional_encoding("nope", head_dim=8)
    with pytest.raises(KeyError, match="registered"):
        build_norm("nope", 8)


def test_rms_norm_matches_formula():
    x = torch.tensor([[3.0, 4.0]])
    out = RMSNorm(2, eps=0.0)(x)
    rms = ((9 + 16) / 2) ** 0.5
    assert torch.allclose(out, x / rms)


def _tiny_cfg(**overrides) -> ModelConfig:
    return ModelConfig(d_model=32, n_heads=4, n_layers=2, d_ff=64, max_length=16, **overrides)


def test_rope_is_the_default_and_adds_no_position_parameters():
    cfg = _tiny_cfg()
    assert cfg.pos_encoding == "rope"
    backbone = build_backbone(50, cfg)
    assert isinstance(backbone, GPTBackbone)
    assert isinstance(backbone.positional, RotaryPositionalEncoding)
    assert not any("table" in n for n, _ in backbone.named_parameters())


@pytest.mark.parametrize("pos", ["rope", "learned"])
@pytest.mark.parametrize("norm", ["layer", "rms"])
@pytest.mark.parametrize("impl", ["mha", "gqa", "mqa"])
def test_backbone_forward_for_every_combination(pos, norm, impl):
    cfg = _tiny_cfg(pos_encoding=pos, norm_type=norm, attn_impl=impl, attn_type="causal")
    backbone = build_backbone(50, cfg)
    tokens = torch.randint(0, 50, (2, 16))
    logits = backbone(tokens)
    assert logits.shape == (2, 16, 50)
    assert torch.isfinite(logits).all()


def test_causality_is_preserved_with_rope():
    backbone = build_backbone(50, _tiny_cfg(attn_type="causal")).eval()
    a = torch.randint(0, 50, (1, 12))
    b = a.clone()
    b[0, 8:] = (b[0, 8:] + 1) % 50  # change only the future
    with torch.no_grad():
        assert torch.allclose(backbone(a)[:, :8], backbone(b)[:, :8], atol=1e-5)


def test_padded_keys_do_not_produce_nan():
    backbone = build_backbone(50, _tiny_cfg(attn_type="causal")).eval()
    tokens = torch.randint(0, 50, (1, 8))
    mask = torch.tensor([[0, 0, 0, 1, 1, 1, 1, 1]])  # left padding: row 0 sees only pads
    assert torch.isfinite(backbone(tokens, attention_mask=mask)).all()


def test_unknown_config_names_fail_at_build_time():
    with pytest.raises(KeyError, match="positional encoding"):
        build_backbone(50, _tiny_cfg(pos_encoding="alibi"))
    with pytest.raises(KeyError, match="norm"):
        build_backbone(50, _tiny_cfg(norm_type="batch"))


def test_lgt_ignores_the_switches_but_still_builds():
    backbone = build_backbone(50, _tiny_cfg(arch="lgt", attn_type="bidirectional", pos_encoding="learned"))
    assert not isinstance(backbone, GPTBackbone)
