"""The transformer block(s) inside each MTP depth are configured like the trunk's, and default to it."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

import hale_llm.models  # noqa: F401
from hale_llm.config.schema import load_config
from hale_llm.config.sections.model import ModelConfig
from hale_llm.nn.moe import auxiliary_loss
from hale_llm.registry import get_loss, get_variant

VOCAB, T = 40, 14


def _cfg(**mtp):
    cfg = load_config("configs/autoregressive_mtp/dense_nano.yaml")
    cfg.set_context_window(T)
    cfg.model.d_model, cfg.model.n_heads, cfg.model.n_layers, cfg.model.d_ff = 32, 4, 2, 64
    cfg.model.dropout, cfg.model.moe_hidden_dim, cfg.model.moe_router = 0.0, 48, "topk"
    cfg.model.mtp_depth = 2
    for key, value in mtp.items():
        setattr(cfg.model, key, value)
    return cfg


def _model(cfg, seed=0):
    torch.manual_seed(seed)
    return get_variant(cfg.variant)(vocab_size=VOCAB, cfg=cfg)


def _params(module) -> int:
    return sum(p.numel() for p in module.parameters())


def test_the_blocks_follow_the_trunk_unless_overridden():
    block = _cfg().model.mtp_block_config()
    assert (block.n_heads, block.ffn_type, block.d_ff, block.d_model) == (4, "mlp", 64, 32)
    block = _cfg(mtp_d_ff=96, mtp_ffn_type="moe").model.mtp_block_config()
    assert (block.n_heads, block.ffn_type, block.d_ff, block.moe_hidden_dim, block.d_model) == (4, "moe", 96, 96, 32)


@pytest.mark.parametrize("layers", [1, 2, 3])
def test_mtp_n_layers_sets_the_number_of_blocks_per_depth(layers):
    model = _model(_cfg(mtp_n_layers=layers))
    assert [len(depth.blocks) for depth in model.mtp.depths] == [layers, layers]


def test_each_extra_block_adds_exactly_one_blocks_worth_of_parameters():
    one, three = _model(_cfg(mtp_n_layers=1)), _model(_cfg(mtp_n_layers=3))
    per_block = _params(one.mtp.depths[0].blocks[0])
    assert _params(three.mtp) - _params(one.mtp) == 2 * 2 * per_block  # 2 depths x 2 extra blocks


def test_the_feed_forward_width_can_differ_from_the_trunk_but_the_heads_cannot():
    model = _model(_cfg(mtp_d_ff=96))
    block = model.mtp.depths[0].blocks[0]
    assert block.ffn.up.weight.shape[0] == 96 and model.backbone.layers[0].ffn.up.weight.shape[0] == 64
    assert len(block.attn.heads) == len(model.backbone.layers[0].attn.heads) == 4  # they share the trunk's rotary positions


def test_the_mtp_blocks_can_be_a_mixture_of_experts_while_the_trunk_is_dense():
    cfg = _cfg(mtp_ffn_type="moe", mtp_d_ff=48)
    model = _model(cfg).train()
    assert type(model.backbone.layers[0].ffn).__name__ == "FeedForward"
    assert type(model.mtp.depths[0].blocks[0].ffn).__name__ == "DeepseekMoE" and model.uses_moe
    ids = torch.randint(0, VOCAB, (2, T))
    labels = ids.clone()
    labels[:, :-1], labels[:, -1] = ids[:, 1:], -100
    loss = get_loss(cfg.variant)(model, {"input_ids": ids, "attention_mask": torch.ones_like(ids), "labels": labels})
    assert auxiliary_loss(model.mtp) > 0 and auxiliary_loss(model.backbone) == 0  # the balancing loss comes from MTP only
    loss.backward()
    expert = model.mtp.depths[0].blocks[0].ffn.routed_experts[0]
    assert expert.w1.weight.shape[0] == 48  # mtp_d_ff is the expert width


def test_invalid_block_settings_are_rejected():
    for bad in ({"mtp_n_layers": 0}, {"mtp_d_ff": 0}):
        with pytest.raises(ValueError):
            ModelConfig(d_model=32, n_heads=4, **bad)
    with pytest.raises(ValueError):
        ModelConfig(mtp_n_heads=2)  # no such option: the heads are the trunk's


# ----- the index conventions do not change when a depth has several blocks or different heads -----
def _depth_logits(model, ids):
    return model.mtp(model.backbone.encode(ids), ids, None, model.backbone.token_emb, model.backbone.lm_head)


@pytest.mark.parametrize("settings", [{"mtp_n_layers": 3}, {"mtp_d_ff": 96}, {"mtp_n_layers": 2, "mtp_ffn_type": "moe"}])
def test_depth_k_position_i_still_sees_exactly_the_tokens_up_to_i_plus_k(settings):
    model = _model(_cfg(**settings)).eval()
    ids = torch.randint(0, VOCAB, (1, T))
    base = _depth_logits(model, ids)
    for k in (1, 2):
        for i in range(T - k):
            if i + k + 1 < T:
                later = ids.clone()
                later[0, i + k + 1 :] = (later[0, i + k + 1 :] + 7) % VOCAB
                assert torch.allclose(_depth_logits(model, later)[k - 1][:, : i + 1], base[k - 1][:, : i + 1], atol=1e-6), (k, i)
            changed = ids.clone()
            changed[0, i + k] = (changed[0, i + k] + 11) % VOCAB
            assert not torch.allclose(_depth_logits(model, changed)[k - 1][:, i], base[k - 1][:, i], atol=1e-6), (k, i)


@pytest.mark.parametrize("settings", [{"mtp_n_layers": 2}, {"mtp_d_ff": 96}])
def test_the_loss_still_equals_the_reference_loop(settings):
    cfg = _cfg(**settings)
    model = _model(cfg).train()
    ids = torch.randint(0, VOCAB, (3, T))
    labels = ids.clone()
    labels[:, :-1], labels[:, -1] = ids[:, 1:], -100
    ours = get_loss(cfg.variant)(model, {"input_ids": ids, "attention_mask": torch.ones_like(ids), "labels": labels})

    backbone = model.backbone  # the reference loop, with the module's blocks applied in turn
    h_prev, mtp_loss = backbone.encode(ids), 0.0
    for k in (1, 2):
        module = model.mtp.depths[k - 1]
        h = module.proj(torch.cat([module.hidden_norm(h_prev[:, :-1]), module.embed_norm(backbone.token_emb(ids[:, k:]))], dim=-1))
        for block in module.blocks:
            h = block(h, torch.arange(h.size(1)))
        mtp_loss = mtp_loss + F.cross_entropy(backbone.lm_head(module.head_norm(h))[:, :-1].reshape(-1, VOCAB), ids[:, k + 1 :].reshape(-1))
        h_prev = h
    main = F.cross_entropy(backbone.lm_head(backbone.encode(ids))[:, :-1].reshape(-1, VOCAB), ids[:, 1:].reshape(-1))
    assert torch.allclose(ours, main + model.mtp_weight * mtp_loss / 2, atol=1e-5)


@pytest.mark.parametrize("settings", [{"mtp_n_layers": 2}, {"mtp_d_ff": 96}])
def test_training_and_inference_still_agree_on_the_same_tokens(settings):
    L, depth = 7, 2
    model = _model(_cfg(**settings)).double().eval()
    ids = torch.arange(L + depth).view(1, -1) % VOCAB
    trained_way = _depth_logits(model, ids)
    previous = model.backbone.encode(ids[:, :L])
    for k in range(1, depth + 1):
        previous, logits = model.mtp.step(k, previous, ids[:, : L + k], model.backbone.token_emb, model.backbone.lm_head)
        assert torch.allclose(logits[0], trained_way[k - 1][0, L - 1], atol=1e-9), f"depth {k}"
