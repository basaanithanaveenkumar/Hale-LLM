"""The dense and mixture-of-experts autoregressive variants, end to end on tiny shapes."""

import math

import pytest
import torch

import hale_llm.models  # noqa: F401
from hale_llm.config.schema import RunConfig
from hale_llm.nn.blocks import TransformerBlock, build_ffn
from hale_llm.nn.moe import auxiliary_loss
from hale_llm.nn.positional import build_positional_encoding
from hale_llm.registry import get_loss, get_sampler, get_variant

VARIANTS = ["autoregressive", "autoregressive_moe"]


def _cfg(variant, **model):
    return RunConfig(
        variant=variant,
        data={"context_window": 16},
        model={"d_model": 32, "n_heads": 4, "n_layers": 2, "d_ff": 64, **model},
    )


def _batch(vocab=50, seq=16):
    ids = torch.randint(0, vocab, (3, seq))
    labels = ids.clone()
    labels[:, :-1] = ids[:, 1:]
    labels[:, -1] = -100
    return {"input_ids": ids, "attention_mask": torch.ones_like(ids), "labels": labels}


@pytest.mark.parametrize("variant", VARIANTS)
def test_variant_trains_one_step_and_samples(variant, tiny_tokenizer):
    cfg = _cfg(variant)
    model = get_variant(variant)(vocab_size=50, cfg=cfg)
    loss = get_loss(variant)(model, _batch())
    loss.backward()
    assert torch.isfinite(loss)
    assert model.backbone.token_emb.weight.grad is not None
    out = get_sampler(variant)(model, tiny_tokenizer, torch.tensor([[5, 6]]), max_new_tokens=4)
    assert out.shape[0] == 1 and out.shape[1] >= 3


def test_moe_variant_forces_moe_and_adds_the_balance_loss():
    dense = get_variant("autoregressive")(vocab_size=50, cfg=_cfg("autoregressive"))
    moe = get_variant("autoregressive_moe")(vocab_size=50, cfg=_cfg("autoregressive_moe", ffn_type="mlp"))  # config overrides to moe
    assert not dense.uses_moe and moe.uses_moe
    assert sum(p.numel() for p in moe.parameters()) > sum(p.numel() for p in dense.parameters())
    assert float(auxiliary_loss(dense)) == 0.0

    batch = _batch()
    moe.train()
    with_aux = float(get_loss("autoregressive_moe")(moe, batch))
    moe.cfg.model.moe_aux_loss_coef = 0.0
    without_aux = float(get_loss("autoregressive_moe")(moe, batch))
    assert with_aux > without_aux


def test_dense_variant_can_switch_to_moe_by_config():
    model = get_variant("autoregressive")(vocab_size=50, cfg=_cfg("autoregressive", ffn_type="moe"))
    assert model.uses_moe


def test_causality():
    model = get_variant("autoregressive")(vocab_size=50, cfg=_cfg("autoregressive")).eval()
    a = _batch()["input_ids"][:1]
    b = a.clone()
    b[0, 10:] = (b[0, 10:] + 1) % 50
    with torch.no_grad():
        la = model({"input_ids": a})[:, :10]
        lb = model({"input_ids": b})[:, :10]
    assert torch.allclose(la, lb, atol=1e-5)


def test_sequences_longer_than_the_context_are_rejected():
    model = get_variant("autoregressive")(vocab_size=50, cfg=_cfg("autoregressive"))
    with pytest.raises(ValueError, match="max_length"):
        model({"input_ids": torch.randint(0, 50, (1, 17))})


def test_block_residual_structure_and_pluggable_ffn():
    cfg = _cfg("autoregressive").model
    block = TransformerBlock(32, 4, ffn=build_ffn("mlp", cfg), causal=True)
    x = torch.randn(2, 5, 32)
    assert block(x).shape == x.shape
    moe_block = TransformerBlock(32, 4, ffn=build_ffn("moe", cfg), positional=build_positional_encoding("rope", head_dim=8))
    assert moe_block(x).shape == x.shape
    assert float(auxiliary_loss(moe_block)) >= 0.0


def test_unknown_ffn_lists_options():
    with pytest.raises(KeyError, match="registered"):
        build_ffn("swiglu", _cfg("autoregressive").model)


def test_model_config_validation():
    with pytest.raises(ValueError, match="divisible"):
        _cfg("autoregressive", d_model=30)
    with pytest.raises(ValueError, match="moe_top_k"):
        _cfg("autoregressive_moe", moe_top_k=9)


def test_moe_variant_sets_ffn_type_in_the_config_so_every_reader_agrees():
    from hale_llm.tokenizer.vocab.param_estimate import estimate_non_embedding_params

    cfg = _cfg("autoregressive_moe", ffn_type="mlp")
    assert cfg.model.ffn_type == "moe"
    dense = _cfg("autoregressive")
    assert estimate_non_embedding_params(cfg.model) > 3 * estimate_non_embedding_params(dense.model)


def test_moe_model_class_rejects_a_dense_config():
    cfg = _cfg("autoregressive")
    with pytest.raises(ValueError, match="ffn_type"):
        get_variant("autoregressive_moe")(vocab_size=50, cfg=cfg)


def test_initialisation_is_gpt_style_and_residual_outputs_are_scaled():
    import math

    model = get_variant("autoregressive_moe")(vocab_size=500, cfg=_cfg("autoregressive_moe", n_layers=8))
    backbone = model.backbone
    assert backbone.token_emb.weight.std().item() == pytest.approx(0.02, rel=0.1)
    assert backbone.lm_head.weight is backbone.token_emb.weight  # tied
    expected = 0.02 / math.sqrt(2 * 8)
    layer = backbone.layers[0]
    assert layer.attn.proj.weight.std().item() == pytest.approx(expected, rel=0.25)
    assert layer.ffn.routed_experts[0].w2.weight.std().item() == pytest.approx(expected, rel=0.25)
    assert layer.attn.proj.bias.abs().max().item() == 0.0


def test_a_fresh_model_scores_close_to_uniform_guessing():
    vocab = 1000
    model = get_variant("autoregressive")(vocab_size=vocab, cfg=_cfg("autoregressive", n_layers=8))
    loss = float(get_loss("autoregressive")(model, _batch(vocab)).detach())
    assert abs(loss - math.log(vocab)) < 0.5
