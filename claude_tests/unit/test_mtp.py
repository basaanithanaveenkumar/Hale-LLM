"""DeepSeek-style multi-token prediction: alignment, the loss, learning, and the plumbing around it."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

import hale_llm.models  # noqa: F401  (registers the variants)
from hale_llm.config.schema import load_config
from hale_llm.nn.moe import auxiliary_loss
from hale_llm.registry import get_loss, get_variant

VOCAB, T = 40, 14


def _cfg(depth=2, weight=0.3, ffn_type="mlp", dropout=0.0):
    cfg = load_config("configs/autoregressive_mtp/dense_nano.yaml")
    cfg.set_context_window(T)
    cfg.model.d_model, cfg.model.n_heads, cfg.model.n_layers, cfg.model.d_ff = 32, 4, 2, 64
    cfg.model.dropout, cfg.model.ffn_type, cfg.model.moe_hidden_dim = dropout, ffn_type, 48
    cfg.model.mtp_depth, cfg.model.mtp_loss_weight = depth, weight
    cfg.model.moe_router = "topk"  # no router noise, so train and eval agree
    return cfg


def _model(cfg, seed=0):
    torch.manual_seed(seed)
    return get_variant(cfg.variant)(vocab_size=VOCAB, cfg=cfg)


def _batch(ids, mask=None):
    labels = ids.clone()
    labels[:, :-1] = ids[:, 1:]
    labels[:, -1] = -100
    if mask is not None:
        labels[mask == 0] = -100  # nothing to predict from a padding position ...
        labels[:, :-1][mask[:, 1:] == 0] = -100  # ... and a padding token is never a target
    return {"input_ids": ids, "attention_mask": mask if mask is not None else torch.ones_like(ids),
            "labels": labels, "pad_token_id": None}


def _depth_logits(model, ids):
    hidden = model.backbone.encode(ids)
    return model.mtp(hidden, ids, None, model.backbone.token_emb, model.backbone.lm_head)


# ----- alignment -----
@pytest.mark.parametrize("depth", [1, 2, 3])
def test_depth_k_position_i_sees_tokens_up_to_i_plus_k_and_no_further(depth):
    model = _model(_cfg(depth)).eval()
    ids = torch.randint(0, VOCAB, (1, T))
    base = _depth_logits(model, ids)
    for k in range(1, depth + 1):
        for i in range(T - k):
            # Changing any token after t_{i+k} must not move the logits at positions <= i.
            if i + k + 1 < T:
                later = ids.clone()
                later[0, i + k + 1 :] = (later[0, i + k + 1 :] + 7) % VOCAB
                assert torch.allclose(_depth_logits(model, later)[k - 1][:, : i + 1], base[k - 1][:, : i + 1], atol=1e-6), (k, i)
            # But t_{i+k} itself must reach position i: it is the embedding the module reads.
            changed = ids.clone()
            changed[0, i + k] = (changed[0, i + k] + 11) % VOCAB
            assert not torch.allclose(_depth_logits(model, changed)[k - 1][:, i], base[k - 1][:, i], atol=1e-6), (k, i)


def test_each_depth_has_the_right_shape():
    model = _model(_cfg(3)).eval()
    logits = _depth_logits(model, torch.randint(0, VOCAB, (2, T)))
    assert [tuple(x.shape) for x in logits] == [(2, T - 1, VOCAB), (2, T - 2, VOCAB), (2, T - 3, VOCAB)]


def test_too_short_a_sequence_is_rejected():
    model = _model(_cfg(3)).eval()
    with pytest.raises(ValueError, match="at least"):
        _depth_logits(model, torch.randint(0, VOCAB, (1, 4)))


# ----- the loss -----
def reference_loss(model, ids, lam, depth):
    """A literal transcription of the reference loop (with the per-depth output norm)."""
    backbone = model.backbone
    h_prev = backbone.encode(ids)
    mtp_loss = 0.0
    for k in range(1, depth + 1):
        module = model.mtp.depths[k - 1]
        e = backbone.token_emb(ids[:, k:])
        h = module.proj(torch.cat([module.hidden_norm(h_prev[:, :-1]), module.embed_norm(e)], dim=-1))
        for block in module.blocks:
            h = block(h, torch.arange(h.size(1)))
        logits = backbone.lm_head(module.head_norm(h))
        mtp_loss = mtp_loss + F.cross_entropy(logits[:, :-1].reshape(-1, VOCAB), ids[:, k + 1 :].reshape(-1))
        h_prev = h
    main = F.cross_entropy(backbone.lm_head(backbone.encode(ids))[:, :-1].reshape(-1, VOCAB), ids[:, 1:].reshape(-1))
    return main + (lam / depth) * mtp_loss


@pytest.mark.parametrize("depth,lam", [(1, 0.3), (2, 0.3), (3, 0.5)])
def test_the_loss_equals_the_reference_loop(depth, lam):
    cfg = _cfg(depth, lam)
    model = _model(cfg).train()
    ids = torch.randint(0, VOCAB, (3, T))
    ours = get_loss(cfg.variant)(model, _batch(ids))
    assert torch.allclose(ours, reference_loss(model, ids, lam, depth), atol=1e-5)


def test_loss_parts_add_up_with_the_weight_over_depth():
    cfg = _cfg(depth=2, weight=0.4)
    model = _model(cfg).train()
    total = get_loss(cfg.variant)(model, _batch(torch.randint(0, VOCAB, (2, T))))
    parts = model.loss_parts
    expected = parts["ntp_loss"] + (0.4 / 2) * (parts["mtp_loss_depth1"] + parts["mtp_loss_depth2"])
    assert torch.allclose(total, expected, atol=1e-6)
    assert torch.allclose(parts["mtp_loss"], (parts["mtp_loss_depth1"] + parts["mtp_loss_depth2"]) / 2, atol=1e-6)
    assert all(not value.requires_grad for value in parts.values())  # pieces are detached


def test_evaluation_scores_the_next_token_loss_only():
    cfg = _cfg()
    model = _model(cfg)
    batch = _batch(torch.randint(0, VOCAB, (2, T)))
    model.train()
    train_total = get_loss(cfg.variant)(model, batch)
    model.eval()
    val = get_loss(cfg.variant)(model, batch)
    assert torch.allclose(val, model.loss_parts["ntp_loss"], atol=1e-6) and val < train_total


def test_a_fresh_model_scores_ln_vocab_times_one_plus_lambda():
    import math

    cfg = _cfg(depth=2, weight=0.3)
    model = _model(cfg).train()
    loss = get_loss(cfg.variant)(model, _batch(torch.randint(0, VOCAB, (8, T))))
    assert abs(float(loss) - math.log(VOCAB) * type(model).initial_loss_scale(cfg)) < 0.15


def test_padding_is_ignored_in_every_depth():
    cfg = _cfg(depth=2)
    model = _model(cfg).train()
    ids = torch.randint(0, VOCAB, (2, T))
    mask = torch.ones_like(ids)
    mask[:, 9:] = 0  # the last 5 tokens are padding
    first = get_loss(cfg.variant)(model, _batch(ids, mask))
    scrambled = ids.clone()
    scrambled[:, 9:] = torch.randint(0, VOCAB, (2, T - 9))  # whatever sits in the padding must not matter
    assert torch.allclose(first, get_loss(cfg.variant)(model, _batch(scrambled, mask)), atol=1e-6)


# ----- sharing, gradients, MoE -----
def test_the_embedding_and_the_head_are_shared_not_copied():
    model = _model(_cfg())
    mtp_params = {id(p) for p in model.mtp.parameters()}
    assert id(model.backbone.token_emb.weight) not in mtp_params and id(model.backbone.lm_head.weight) not in mtp_params
    assert model.backbone.lm_head.weight is model.backbone.token_emb.weight  # tied, as in the trunk
    names = {name for name, _ in model.mtp.named_parameters()}
    assert {"depths.0.proj.weight", "depths.0.hidden_norm.weight", "depths.0.embed_norm.weight", "depths.0.head_norm.weight"} <= names


def test_every_part_of_the_model_gets_a_gradient():
    cfg = _cfg(depth=2)
    model = _model(cfg).train()
    get_loss(cfg.variant)(model, _batch(torch.randint(0, VOCAB, (2, T)))).backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
        assert parameter.grad.abs().sum() > 0, f"{name} got an all-zero gradient"


def test_moe_trunk_and_moe_mtp_blocks_both_feed_the_balancing_loss():
    cfg = _cfg(depth=2, ffn_type="moe")
    model = _model(cfg).train()
    assert type(model.mtp.depths[0].blocks[0].ffn).__name__ == "DeepseekMoE"
    loss = get_loss(cfg.variant)(model, _batch(torch.randint(0, VOCAB, (2, T))))
    mtp_aux = auxiliary_loss(model.mtp)
    assert mtp_aux > 0 and torch.allclose(auxiliary_loss(model), auxiliary_loss(model.backbone) + mtp_aux)
    loss.backward()


# ----- config -----
def test_hyperparameters_come_from_the_config():
    cfg = load_config("configs/autoregressive_mtp/dense_nano.yaml")
    m = cfg.model  # the defaults: 1 MTP depth (2 tokens per position), DeepSeek-V3's lambda schedule, one block per depth
    assert (m.mtp_depth, m.mtp_loss_weight, m.mtp_loss_weight_final, m.mtp_switch_fraction) == (1, 0.3, 0.1, 0.68)
    assert (m.mtp_n_layers, m.mtp_d_ff, m.mtp_ffn_type) == (1, None, None)
    assert cfg.sample.speculative is True
    assert len(_model(_cfg(depth=3)).mtp.depths) == 3  # D is a config value, not a constant


def test_invalid_mtp_settings_are_rejected():
    from hale_llm.config.sections.model import ModelConfig

    with pytest.raises(ValueError):
        ModelConfig(mtp_depth=0)
    with pytest.raises(ValueError):
        ModelConfig(mtp_loss_weight=-0.1)


# ----- it learns -----
def test_the_depth_heads_learn_to_predict_further_ahead():
    """On a periodic sequence every depth should become (nearly) perfect: alignment and learnability together."""
    cfg = _cfg(depth=2, weight=1.0)
    cfg.model.d_model, cfg.model.n_layers = 48, 2
    model = _model(cfg).train()
    period = 7
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-3)
    generator = torch.Generator().manual_seed(0)
    for _ in range(250):
        offsets = torch.randint(0, period, (16, 1), generator=generator)
        ids = (offsets + torch.arange(T)) % period
        optimizer.zero_grad()
        get_loss(cfg.variant)(model, _batch(ids)).backward()
        optimizer.step()
    model.eval()
    ids = (torch.randint(0, period, (32, 1), generator=generator) + torch.arange(T)) % period
    with torch.no_grad():
        for k, logits in enumerate(_depth_logits(model, ids), start=1):
            predicted = logits[:, :-1].argmax(-1)  # position j predicts t_{j+k+1}
            accuracy = (predicted == ids[:, k + 1 :]).float().mean().item()
            assert accuracy > 0.95, f"depth {k} accuracy {accuracy:.2f}"


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_the_mtp_blocks_keep_the_residual_stream_in_the_trunks_precision_under_autocast(dtype):
    """The trunk's residual stream stays fp32 under autocast; the MTP blocks must not quietly drop to half precision."""
    cfg = _cfg(depth=1)
    model = _model(cfg).train()
    streams: dict[str, torch.dtype] = {}
    model.backbone.layers[0].register_forward_hook(lambda m, a, o: streams.update(trunk=o.dtype))
    model.mtp.depths[0].blocks[0].register_forward_hook(lambda m, a, o: streams.update(mtp=o.dtype))
    ids = torch.randint(0, VOCAB, (2, T))
    with torch.autocast("cpu", dtype=dtype):
        loss = get_loss(cfg.variant)(model, _batch(ids))
    assert streams == {"trunk": torch.float32, "mtp": torch.float32} and torch.isfinite(loss)
