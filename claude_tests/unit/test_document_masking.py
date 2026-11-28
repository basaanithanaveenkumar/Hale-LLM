"""Document masking (model.document_masking): the mask, the attention paths, isolation, edge cases and gradients."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

import hale_llm.models  # noqa: F401
from hale_llm.config.schema import load_config
from hale_llm.config.sections.model import ModelConfig
from hale_llm.dataloader.document_ids import document_ids_from_eos
from hale_llm.models.models_autoregressive.model import collate_autoregressive
from hale_llm.nn.attention import HeadAttention, cross_document_mask
from hale_llm.registry import get_loss, get_variant

EOS, VOCAB, T = 0, 40, 16  # token 0 ends a document; the window below holds three documents


def _window(eos_at=(5, 10), seed=0) -> torch.Tensor:
    """[1, 16] tokens with the end-of-text token at the given positions: documents are 0..5, 6..10 and 11..15."""
    ids = torch.randint(1, VOCAB, (1, T), generator=torch.Generator().manual_seed(seed))
    for position in eos_at:
        ids[0, position] = EOS
    return ids


def _tokenizer(eos_id=EOS):
    return type("Tok", (), {"eos_token_id": eos_id, "pad_token_id": None})()


def _batch(ids: torch.Tensor, masked: bool) -> dict:
    return collate_autoregressive(ids, _tokenizer(), document_masking=masked)


def _model(variant="autoregressive", ffn_type="mlp", depth=2, seed=0, dropout=0.0):
    cfg = load_config("configs/autoregressive_mtp/dense_nano.yaml")
    cfg.variant = variant
    cfg.set_context_window(T)
    cfg.model.d_model, cfg.model.n_heads, cfg.model.n_layers, cfg.model.d_ff = 32, 4, 2, 64
    cfg.model.dropout, cfg.model.ffn_type, cfg.model.moe_hidden_dim, cfg.model.moe_router = dropout, ffn_type, 48, "topk"
    cfg.model.mtp_depth, cfg.model.document_masking = depth, True
    torch.manual_seed(seed)
    return get_variant(variant)(vocab_size=VOCAB, cfg=cfg)


VARIANTS = [("autoregressive", "mlp"), ("autoregressive_moe", "moe"), ("autoregressive_mtp", "mlp"), ("autoregressive_moe_mtp", "moe")]


# ----- the document ids -----
@pytest.mark.parametrize(
    "ids,expected",
    [
        ([4, 5, 9, 7, 8, 9, 3], [0, 0, 0, 1, 1, 1, 2]),  # the documented example (end-of-text = 9)
        ([1, 2, 3], [0, 0, 0]),  # no end-of-text: one document
        ([1, 2, 9], [0, 0, 0]),  # ends with the token: it belongs to the document it ends
        ([9, 1, 2], [0, 1, 1]),  # starts with it: a one-token document, then a new one
        ([9, 9, 1], [0, 1, 2]),  # consecutive end-of-text tokens: an empty document in between
        ([9, 9, 9], [0, 1, 2]),  # nothing but separators
        ([9], [0]),
    ],
)
def test_document_ids_follow_the_end_of_text_tokens(ids, expected):
    assert document_ids_from_eos(torch.tensor([ids]), eos_id=9).tolist() == [expected]


def test_rows_of_a_batch_are_numbered_independently():
    ids = torch.tensor([[1, 9, 2, 9], [3, 3, 3, 9]])
    assert document_ids_from_eos(ids, 9).tolist() == [[0, 0, 1, 1], [0, 0, 0, 0]]


def test_the_cross_document_mask_blocks_exactly_the_pairs_from_different_documents():
    mask = cross_document_mask(torch.tensor([[0, 0, 1, 1, 2]]))
    assert mask.shape == (1, 5, 5) and not mask[0].diagonal().any()  # every token sees itself
    assert torch.equal(mask, mask.transpose(1, 2))  # symmetric: the causal triangle is applied by the head
    assert mask[0, 0, 1].item() is False and mask[0, 1, 2].item() is True and mask[0, 3, 4].item() is True
    assert mask[0].sum().item() == 25 - (2 * 2 + 2 * 2 + 1)  # 25 pairs minus those inside a document


# ----- the attention head: fused-with-mask path against the explicit path -----
def _head(seed=0, dropout=0.0):
    torch.manual_seed(seed)
    return HeadAttention(16, 8, dropout=dropout, causal=True)


def test_a_mask_with_no_boundaries_equals_the_plain_causal_fast_path():
    head, x = _head().eval(), torch.randn(2, 12, 16)
    none_blocked = cross_document_mask(torch.zeros(2, 12, dtype=torch.long))
    assert torch.allclose(head(x), head(x, document_mask=none_blocked), atol=1e-6)


def test_the_fused_masked_path_equals_the_explicit_path_and_gives_zero_weight_across_documents():
    x = torch.randn(2, 12, 16)
    ids = torch.tensor([[0, 0, 0, 1, 1, 2, 2, 2, 2, 3, 3, 3], [0] * 12])
    mask = cross_document_mask(ids)
    fused = _head().eval()
    explicit = _head().eval()
    explicit.return_attn_weights = True  # forces the explicit softmax path
    out_fused = fused(x, document_mask=mask)
    out_explicit, weights = explicit(x, document_mask=mask)
    assert torch.allclose(out_fused, out_explicit, atol=1e-5)
    assert torch.allclose(weights.sum(-1), torch.ones(2, 12), atol=1e-5)  # still a distribution
    assert weights[mask.expand_as(weights)].abs().max() == 0  # nothing leaks across documents
    assert weights.triu(1).abs().max() == 0  # nor from the future


def test_padding_and_documents_can_be_combined():
    x = torch.randn(1, 10, 16)
    mask = cross_document_mask(torch.tensor([[0] * 4 + [1] * 6]))
    padding = torch.zeros(1, 10, dtype=torch.bool)
    padding[0, 8:] = True
    head = _head().eval()
    head.return_attn_weights = True
    out, weights = head(x, key_padding_mask=padding, document_mask=mask)
    assert torch.isfinite(out).all() and weights[0, :, 8:].abs().max() == 0 and weights[0, 5:, :4].abs().max() == 0


def test_dropout_in_training_mode_still_runs_and_is_finite():
    head = _head(dropout=0.3).train()
    out = head(torch.randn(2, 12, 16), document_mask=cross_document_mask(torch.tensor([[0] * 6 + [1] * 6] * 2)))
    assert out.shape == (2, 12, 8) and torch.isfinite(out).all()


# ----- the models: documents do not see each other, and each behaves as if it were alone -----
@pytest.mark.parametrize("variant,ffn", VARIANTS)
def test_changing_an_earlier_document_does_not_change_later_ones_when_masked_but_does_when_not(variant, ffn):
    model = _model(variant, ffn).eval()
    ids = _window()
    other = ids.clone()
    other[0, :5] = (other[0, :5] % (VOCAB - 1)) + 1  # rewrite most of the first document (positions 0..4)
    other[0, 0] = (ids[0, 0] % (VOCAB - 1)) + 1
    assert not torch.equal(ids, other) and other[0, 5] == EOS  # the separator stays where it is
    with torch.no_grad():
        masked = model(_batch(ids, True))[:, 6:], model(_batch(other, True))[:, 6:]
        plain = model(_batch(ids, False))[:, 6:], model(_batch(other, False))[:, 6:]
    assert torch.allclose(masked[0], masked[1], atol=1e-6)  # isolated
    assert (plain[0] - plain[1]).abs().max() > 1e-4  # the control: without the mask the first document leaks in


@pytest.mark.parametrize("variant,ffn", VARIANTS)
def test_isolation_also_holds_when_the_trunk_is_called_without_a_padding_mask(variant, ffn):
    """The fused attention path is used when there is no padding mask: it must honour the document mask too."""
    model = _model(variant, ffn).eval()
    ids = _window()
    other = ids.clone()
    other[0, 1:5] = (other[0, 1:5] % (VOCAB - 1)) + 1
    doc = document_ids_from_eos(ids, EOS)
    with torch.no_grad():
        a = model.backbone.encode(ids, document_ids=doc)[:, 6:]
        b = model.backbone.encode(other, document_ids=doc)[:, 6:]
        leaky = model.backbone.encode(ids)[:, 6:], model.backbone.encode(other)[:, 6:]
        masked_padded = model.backbone.encode(ids, attention_mask=torch.ones_like(ids), document_ids=doc)[:, 6:]
    assert torch.allclose(a, b, atol=1e-6) and (leaky[0] - leaky[1]).abs().max() > 1e-4
    assert torch.allclose(a, masked_padded, atol=1e-5)  # with and without the (all-ones) padding mask: the same answer


@pytest.mark.parametrize("variant,ffn", VARIANTS)
def test_each_document_in_a_window_behaves_exactly_as_if_it_were_alone(variant, ffn):
    """With rotary positions only relative distances matter, so a document packed at an offset must give the same logits as alone."""
    model = _model(variant, ffn).eval()
    ids = _window()
    with torch.no_grad():
        packed = model(_batch(ids, True))
        for start, end in [(0, 6), (6, 11), (11, 16)]:
            alone = model(_batch(ids[:, start:end], False))
            assert torch.allclose(packed[:, start:end], alone, atol=1e-5), (variant, start, end)


@pytest.mark.parametrize("variant,ffn", VARIANTS)
def test_attention_inside_a_document_is_still_causal(variant, ffn):
    model = _model(variant, ffn).eval()
    ids = _window()
    later = ids.clone()
    later[0, 8] = (later[0, 8] % (VOCAB - 1)) + 1
    with torch.no_grad():
        a, b = model(_batch(ids, True)), model(_batch(later, True))
    assert torch.allclose(a[:, :8], b[:, :8], atol=1e-6) and (a[:, 8:11] - b[:, 8:11]).abs().max() > 1e-4


@pytest.mark.parametrize("variant,ffn", [("autoregressive_mtp", "mlp"), ("autoregressive_moe_mtp", "moe")])
def test_the_mtp_depths_respect_the_documents_too(variant, ffn):
    model = _model(variant, ffn, depth=2).eval()
    ids = _window()
    other = ids.clone()
    other[0, 1:5] = (other[0, 1:5] % (VOCAB - 1)) + 1
    bb = model.backbone

    def depth_logits(tokens, masked):
        batch = _batch(tokens, masked)
        hidden = bb.encode(tokens, document_ids=batch.get("document_ids"))
        return model.mtp(hidden, tokens, None, bb.token_emb, bb.lm_head, batch.get("document_ids"))

    with torch.no_grad():
        a, b = depth_logits(ids, True), depth_logits(other, True)
        control = depth_logits(ids, False), depth_logits(other, False)
        for k in (1, 2):  # positions of the second document (6..10) at depth k, wherever their tokens stay in the same document
            assert torch.allclose(a[k - 1][:, 6 : 6 + 5 - k], b[k - 1][:, 6 : 6 + 5 - k], atol=1e-6), k
        assert (control[0][0][:, 6:10] - control[1][0][:, 6:10]).abs().max() > 1e-4
        # and a document at depth k equals the same document alone, at the positions whose tokens stay inside it
        for k in (1, 2):
            alone_tokens = ids[:, 6:11]
            alone = depth_logits(alone_tokens, False)[k - 1]
            assert torch.allclose(a[k - 1][:, 6 : 6 + 5 - k], alone[:, : 5 - k], atol=1e-5), k


# ----- backward: gradients are the gradients of the documents run separately -----
def _sum_nll(model, ids, mask_docs, skip_separator_targets=True):
    """Summed next-token loss over a window. The target after a separator is the next document's first token, which
    exists only when documents are packed, so it is left out here and in the separate runs: the two are comparable."""
    batch = _batch(ids, mask_docs)
    labels = batch["labels"].clone()
    if skip_separator_targets:
        labels[ids == EOS] = -100
    logits = model(batch)
    return F.cross_entropy(logits[:, :-1].reshape(-1, VOCAB), labels[:, :-1].reshape(-1), ignore_index=-100, reduction="sum")


@pytest.mark.parametrize("variant,ffn", [("autoregressive", "mlp"), ("autoregressive_moe", "moe")])
def test_the_gradients_of_a_packed_masked_window_equal_the_gradients_of_its_documents_run_separately(variant, ffn):
    ids = _window()
    packed_model = _model(variant, ffn).eval()
    _sum_nll(packed_model, ids, True).backward()
    separate_model = _model(variant, ffn).eval()  # same seed: same weights
    for start, end in [(0, 6), (6, 11), (11, 16)]:
        _sum_nll(separate_model, ids[:, start:end], False).backward()  # gradients accumulate over the documents
    packed, separate = dict(packed_model.named_parameters()), dict(separate_model.named_parameters())
    checked = 0
    for name, parameter in packed.items():
        assert parameter.grad is not None and separate[name].grad is not None, name
        assert torch.allclose(parameter.grad, separate[name].grad, atol=1e-5, rtol=1e-4), name
        checked += 1
    assert checked > 10 and any(p.grad.abs().sum() > 0 for p in packed.values())


def test_without_the_mask_those_gradients_differ_which_proves_the_test_can_tell():
    ids = _window()
    packed_model = _model().eval()
    _sum_nll(packed_model, ids, False).backward()  # documents see each other
    separate_model = _model().eval()
    for start, end in [(0, 6), (6, 11), (11, 16)]:
        _sum_nll(separate_model, ids[:, start:end], False).backward()
    packed, separate = dict(packed_model.named_parameters()), dict(separate_model.named_parameters())
    assert any(not torch.allclose(packed[n].grad, separate[n].grad, atol=1e-5, rtol=1e-4) for n in packed)


EDGE_WINDOWS = {
    "no separator": (),
    "separator first": (0,),
    "separator last": (T - 1,),
    "first and last": (0, T - 1),
    "consecutive separators": (4, 5, 6),
    "one-token documents": tuple(range(0, T, 2)),
    "nothing but separators": tuple(range(T)),
    "single separator in the middle": (8,),
}


@pytest.mark.parametrize("variant,ffn", VARIANTS)
@pytest.mark.parametrize("name", list(EDGE_WINDOWS))
def test_edge_case_windows_give_a_finite_loss_and_finite_nonzero_gradients(variant, ffn, name):
    model = _model(variant, ffn, dropout=0.1).train()
    ids = _window(EDGE_WINDOWS[name]).repeat(2, 1)
    loss = get_loss(variant)(model, _batch(ids, True))
    loss.backward()
    assert torch.isfinite(loss), name
    for parameter_name, parameter in model.named_parameters():
        if parameter.grad is None:  # a routed expert that no token chose gets no gradient: normal for a mixture of experts
            assert "routed_experts" in parameter_name, (name, parameter_name)
            continue
        assert torch.isfinite(parameter.grad).all(), (name, parameter_name)
    assert sum(float(p.grad.abs().sum()) for p in model.parameters() if p.grad is not None) > 0


def test_the_registered_loss_with_masking_on_a_window_without_separators_equals_masking_off():
    ids = _window(eos_at=())
    on, off = _model().train(), _model().train()
    assert torch.allclose(get_loss("autoregressive")(on, _batch(ids, True)), get_loss("autoregressive")(off, _batch(ids, False)), atol=1e-6)


# ----- the flag and the plumbing -----
def test_the_flag_is_off_by_default_and_the_batch_then_has_no_document_ids():
    assert ModelConfig().document_masking is False
    batch = collate_autoregressive(_window(), _tokenizer())
    assert "document_ids" not in batch


def test_with_the_flag_the_batch_carries_ids_that_match_the_end_of_text_positions():
    batch = collate_autoregressive(_window(), _tokenizer(), document_masking=True)
    assert batch["document_ids"].tolist() == [[0] * 6 + [1] * 5 + [2] * 5]


def test_the_flag_needs_a_tokenizer_with_an_end_of_text_token():
    with pytest.raises(ValueError, match="end-of-text"):
        collate_autoregressive(_window(), _tokenizer(eos_id=None), document_masking=True)


def test_a_batch_without_document_ids_behaves_exactly_as_before_whatever_the_config_says():
    model = _model().eval()
    ids = _window()
    with torch.no_grad():
        assert torch.equal(model(_batch(ids, False)), model({"input_ids": ids, "attention_mask": torch.ones_like(ids)}))
        assert (model(_batch(ids, True)) - model(_batch(ids, False))).abs().max() > 1e-4  # masking changes the result when there are separators


# ----- litmus: a copy task that only works if documents can see each other -----
def _copy_window_batch(batch_size: int, seed: int, masked: bool) -> dict:
    """Window = [doc A (7 random tokens), EOS, doc B = the same 7 tokens, EOS]. B is predictable only by looking at A."""
    generator = torch.Generator().manual_seed(seed)
    doc = torch.randint(1, VOCAB, (batch_size, 7), generator=generator)
    eos = torch.zeros(batch_size, 1, dtype=torch.long)
    return _batch(torch.cat([doc, eos, doc, eos], dim=1), masked)  # [B, 16]


def _copy_loss_after_training(masked: bool) -> float:
    """Train on the copy task, then return the loss on the SECOND document's tokens only."""
    model = _model("autoregressive", dropout=0.0, seed=3)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
    for step in range(250):
        batch = _copy_window_batch(32, seed=step, masked=masked)
        loss = get_loss("autoregressive")(model, batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    model.eval()
    batch = _copy_window_batch(256, seed=10_000, masked=masked)
    with torch.no_grad():
        logits = model(batch)
    second_doc = slice(8, 15)  # positions whose next token is a copied token of document B
    return F.cross_entropy(logits[:, second_doc].reshape(-1, VOCAB), batch["labels"][:, second_doc].reshape(-1)).item()


def test_litmus_the_model_learns_to_copy_across_documents_only_without_the_mask():
    unmasked, masked = _copy_loss_after_training(False), _copy_loss_after_training(True)
    chance = torch.log(torch.tensor(float(VOCAB - 1))).item()  # uniform guess over the random tokens
    assert unmasked < 0.5 * chance, unmasked  # without the mask it reads document A: the copy is learned
    assert masked > 0.8 * chance, masked  # with the mask document A is invisible: stuck near chance
