"""Offset-conditioned MTP: the balanced sampler, the alignment, V3 depth-1 equivalence, leakage, masking, gradients,
resuming and speculative decoding."""

from __future__ import annotations

import collections
import itertools

import pytest
import torch
import torch.nn.functional as F

import hale_llm.models  # noqa: F401  (registers the variants)
from hale_llm.config.schema import load_config
from hale_llm.models.models_mtp.offset_loss import offset_conditioned_mtp_loss
from hale_llm.models.models_mtp.speculative import SpeculativeDecoder
from hale_llm.nn.mtp import BalancedOffsetSampler, MultiTokenPredictor, OffsetConditionedPredictor
from hale_llm.optim import build_optimizer
from hale_llm.optim.parameter_groups import split_parameters
from hale_llm.registry import get_loss, get_variant

VOCAB, T, D = 40, 14, 32
NO_EOS = type("Tok", (), {"eos_token_id": None})()


# ----------------------------------------------------------------------------- helpers
def _cfg(depth=4, offset=True, draft=1, weight=0.3, ffn_type="mlp", n_layers=1, max_length=T, variant="autoregressive_mtp"):
    cfg = load_config("configs/autoregressive_mtp/dense_nano.yaml")
    cfg.variant = variant
    cfg.set_context_window(max_length)
    m = cfg.model
    m.d_model, m.n_heads, m.n_layers, m.d_ff, m.dropout = D, 4, 2, 64, 0.0
    m.ffn_type, m.moe_hidden_dim, m.moe_router = ffn_type, 48, "topk"
    m.mtp_depth, m.mtp_offset_conditioning, m.mtp_draft_tokens = depth, offset, draft
    m.mtp_loss_weight, m.mtp_n_layers = weight, n_layers
    return cfg


def _model(cfg=None, seed=0, **kwargs):
    cfg = cfg or _cfg(**kwargs)
    torch.manual_seed(seed)
    return get_variant(cfg.variant)(vocab_size=VOCAB, cfg=cfg)


def _batch(ids, mask=None, document_ids=None):
    labels = ids.clone()
    labels[:, :-1] = ids[:, 1:]
    labels[:, -1] = -100
    if mask is not None:
        labels[mask == 0] = -100
        labels[:, :-1][mask[:, 1:] == 0] = -100
    batch = {"input_ids": ids, "attention_mask": mask if mask is not None else torch.ones_like(ids),
             "labels": labels, "pad_token_id": None}
    if document_ids is not None:
        batch["document_ids"] = document_ids
    return batch


def _random_ids(batch_size=4, length=T, seed=0):
    return torch.randint(1, VOCAB, (batch_size, length), generator=torch.Generator().manual_seed(seed))


def _force_offsets(model, values):
    """Make the sampler hand out ``values`` in a cycle (instead of the balanced random stream)."""
    cycle = itertools.cycle(values)
    model.mtp.sampler.sample = lambda count, device="cpu": torch.tensor([next(cycle) for _ in range(count)], device=device)


class _Recorder(torch.nn.Module):
    """A stand-in for the token embedding: remembers the ids it was asked about and returns zeros."""

    def __init__(self):
        super().__init__()
        self.seen = None

    def forward(self, ids):
        self.seen = ids.clone()
        return torch.zeros(*ids.shape, D)


# ----------------------------------------------------------------------------- the sampler
def test_offsets_are_integers_in_range():
    sampler = BalancedOffsetSampler(4, seed=3)
    drawn = sampler.sample(1000)
    assert drawn.dtype == torch.long and drawn.shape == (1000,) and 1 <= int(drawn.min()) and int(drawn.max()) <= 4


@pytest.mark.parametrize("n_offsets", [1, 2, 3, 4, 5, 6])
def test_every_aligned_block_of_n_draws_uses_each_offset_exactly_once(n_offsets):
    drawn = BalancedOffsetSampler(n_offsets, seed=5).sample(n_offsets * 200).view(200, n_offsets)
    assert all(sorted(row.tolist()) == list(range(1, n_offsets + 1)) for row in drawn)


@pytest.mark.parametrize("n_offsets", [2, 4, 5])
def test_after_any_number_of_draws_the_counts_differ_by_at_most_one(n_offsets):
    sampler = BalancedOffsetSampler(n_offsets, seed=11)
    counts = collections.Counter()
    rng = torch.Generator().manual_seed(0)
    for _ in range(300):  # irregular batch sizes, as with different batch sizes and accumulation
        counts.update(sampler.sample(int(torch.randint(1, 13, (1,), generator=rng))).tolist())
        per_offset = [counts[n] for n in range(1, n_offsets + 1)]
        assert max(per_offset) - min(per_offset) <= 1, per_offset


def test_equal_frequency_over_a_whole_run_even_when_the_batch_size_does_not_divide_n():
    sampler = BalancedOffsetSampler(4, seed=2)
    steps, batch = 90, 6  # 540 draws = 135 decks
    counts = collections.Counter(offset for _ in range(steps) for offset in sampler.sample(batch).tolist())
    assert [counts[n] for n in (1, 2, 3, 4)] == [135] * 4


@pytest.mark.parametrize("split", [[25], [10, 15], [1] * 25, [3, 7, 5, 9, 1], [4, 4, 4, 4, 4, 5]])
def test_the_stream_does_not_depend_on_how_the_draws_are_split_into_batches(split):
    whole = BalancedOffsetSampler(4, seed=9).sample(25).tolist()
    sampler = BalancedOffsetSampler(4, seed=9)
    assert [offset for size in split for offset in sampler.sample(size).tolist()] == whole


def test_same_seed_same_order_different_seed_different_order():
    a, b = BalancedOffsetSampler(4, seed=1).sample(400), BalancedOffsetSampler(4, seed=1).sample(400)
    other = BalancedOffsetSampler(4, seed=2).sample(400)
    assert torch.equal(a, b) and not torch.equal(a, other)


def test_the_order_inside_a_deck_is_random_not_a_fixed_cycle():
    decks = BalancedOffsetSampler(4, seed=7).sample(4 * 4000).view(4000, 4)
    assert len({tuple(row.tolist()) for row in decks}) == 24  # all 4! orders occur
    first = collections.Counter(decks[:, 0].tolist())
    assert all(800 < first[n] < 1200 for n in (1, 2, 3, 4)), first  # each offset opens a deck about a quarter of the time
    last = collections.Counter(decks[:, -1].tolist())
    assert all(800 < last[n] < 1200 for n in (1, 2, 3, 4)), last


def test_sampling_does_not_touch_the_global_random_state():
    state = torch.get_rng_state()
    BalancedOffsetSampler(4, seed=0).sample(1000)
    assert torch.equal(state, torch.get_rng_state())


def test_the_stream_position_is_saved_and_restored_and_a_foreign_state_is_refused():
    sampler = BalancedOffsetSampler(4, seed=6)
    sampler.sample(13)
    saved = sampler.get_extra_state()
    expected_next = sampler.sample(20).tolist()
    restored = BalancedOffsetSampler(4, seed=6)
    restored.set_extra_state(saved)
    assert restored.sample(20).tolist() == expected_next and restored.cursor == 33
    with pytest.raises(ValueError, match="offset sampler state is for"):
        BalancedOffsetSampler(4, seed=7).set_extra_state(saved)
    with pytest.raises(ValueError, match="offset sampler state is for"):
        BalancedOffsetSampler(3, seed=6).set_extra_state(saved)


def test_invalid_sampler_arguments_are_refused():
    with pytest.raises(ValueError, match="at least 1"):
        BalancedOffsetSampler(0)
    with pytest.raises(ValueError, match="seed"):
        BalancedOffsetSampler(4, seed=-1)
    with pytest.raises(ValueError, match="seed"):
        BalancedOffsetSampler(4, seed=2**31)
    with pytest.raises(ValueError, match="negative"):
        BalancedOffsetSampler(4).sample(-1)
    assert BalancedOffsetSampler(4).sample(0).numel() == 0


# ----------------------------------------------------------------------------- the alignment
def test_the_cat_sat_on_the_mat_example():
    """'the cat sat on the mat' with t at 'the': n=1 reads cat and predicts sat, n=2 reads sat -> on, n=3 reads on -> the."""
    the, cat, sat, on, mat = 1, 2, 3, 4, 5
    ids = torch.tensor([[the, cat, sat, on, the, mat]]).repeat(3, 1)  # T = 6, one copy per offset
    offsets = torch.tensor([1, 2, 3])
    predictor = OffsetConditionedPredictor(_cfg(depth=4, max_length=6).model, positional=None)
    recorder = _Recorder()
    predictor(torch.zeros(3, 6, D), ids, None, recorder, torch.nn.Linear(D, VOCAB), offsets)
    targets = predictor.targets(_batch(ids)["labels"], offsets)
    assert recorder.seen[:, 0].tolist() == [cat, sat, on]  # the input tokens at t = 0
    assert targets[:, 0].tolist() == [sat, on, the]  # the targets at t = 0


@pytest.mark.parametrize("positional", [None, "rope"])
def test_input_is_x_t_plus_n_and_target_is_x_t_plus_n_plus_1_for_every_position_and_offset(positional):
    """Identity tokens (token id = 100 + position) make every index visible."""
    length = 12
    ids = (100 + torch.arange(length)).repeat(4, 1)
    offsets = torch.tensor([1, 2, 3, 4])
    cfg = _cfg(depth=4, max_length=length)
    model = _model(cfg)
    predictor = model.mtp
    recorder = _Recorder()
    predictor(torch.zeros(4, length, D), ids, None, recorder, torch.nn.Linear(D, VOCAB), offsets)
    targets = predictor.targets(_batch(ids)["labels"], offsets)
    for b, n in enumerate(offsets.tolist()):
        for t in range(length):
            if t + n + 1 <= length - 1:  # the target exists
                assert recorder.seen[b, t] == 100 + t + n, (b, t)  # input x(t + n)
                assert targets[b, t] == 100 + t + n + 1, (b, t)  # target x(t + n + 1)
            else:
                assert targets[b, t] == -100, (b, t)  # past the end: ignored
    assert (targets != -100).sum(1).tolist() == [length - n - 1 for n in (1, 2, 3, 4)]


def test_offset_one_is_exactly_deepseek_depth_one():
    """Same trunk, same module weights, zero offset embedding, offset 1 everywhere: same logits and the same loss."""
    deepseek = _model(_cfg(depth=1, offset=False), seed=1).train()
    offset = _model(_cfg(depth=4, offset=True), seed=2).train()
    offset.backbone.load_state_dict(deepseek.backbone.state_dict())
    offset.mtp.mtp_module.load_state_dict(deepseek.mtp.depths[0].state_dict())
    torch.nn.init.zeros_(offset.mtp.offset_emb.weight)  # no offset signal left: n = 1 is plain depth 1
    _force_offsets(offset, [1])
    batch = _batch(_random_ids(5, seed=3))
    hidden = deepseek.backbone.encode(batch["input_ids"], attention_mask=batch["attention_mask"])
    depth_logits = deepseek.mtp(hidden, batch["input_ids"], batch["attention_mask"], deepseek.backbone.token_emb,
                                deepseek.backbone.lm_head)[0]  # [B, T-1, V]
    offset_logits = offset.mtp(hidden, batch["input_ids"], batch["attention_mask"], offset.backbone.token_emb,
                               offset.backbone.lm_head, torch.ones(5, dtype=torch.long))
    assert torch.allclose(offset_logits[:, : T - 1], depth_logits, atol=1e-6)
    loss_deepseek, loss_offset = get_loss("autoregressive_mtp")(deepseek, batch), get_loss("autoregressive_mtp")(offset, batch)
    assert torch.allclose(loss_deepseek, loss_offset, atol=1e-6)
    assert torch.allclose(deepseek.loss_parts["mtp_loss"], offset.loss_parts["mtp_loss"], atol=1e-6)


def test_each_sequence_of_a_mixed_batch_behaves_as_if_it_ran_alone_with_its_own_offset():
    model = _model().eval()
    ids = _random_ids(4, seed=4)
    offsets = torch.tensor([1, 3, 2, 4])
    bb, mtp = model.backbone, model.mtp
    with torch.no_grad():
        hidden = bb.encode(ids)
        together = mtp(hidden, ids, None, bb.token_emb, bb.lm_head, offsets)
        for b in range(4):
            alone_hidden = bb.encode(ids[b : b + 1])
            alone = mtp(alone_hidden, ids[b : b + 1], None, bb.token_emb, bb.lm_head, offsets[b : b + 1])
            assert torch.allclose(together[b : b + 1], alone, atol=1e-5), b


def test_the_loss_equals_a_reference_written_with_plain_loops():
    model = _model(weight=0.3).train()
    ids = _random_ids(6, seed=5)
    batch = _batch(ids)
    offsets = [1, 2, 3, 4, 2, 1]
    _force_offsets(model, offsets)
    bb = model.backbone
    with torch.no_grad():
        hidden = bb.encode(ids)
        logits = model.mtp(hidden, ids, None, bb.token_emb, bb.lm_head, torch.tensor(offsets))
        total, count = 0.0, 0
        for b, n in enumerate(offsets):
            for t in range(T):
                if t + n + 1 <= T - 1:  # the module reads x(t+n) and predicts x(t+n+1)
                    total += float(F.cross_entropy(logits[b, t][None], ids[b, t + n + 1][None], reduction="sum"))
                    count += 1
        ntp = F.cross_entropy(bb.lm_head(hidden)[:, :-1].reshape(-1, VOCAB), ids[:, 1:].reshape(-1))
        loss = get_loss("autoregressive_mtp")(model, batch)
    assert float(model.loss_parts["mtp_loss"]) == pytest.approx(total / count, rel=1e-5)
    assert float(loss) == pytest.approx(float(ntp) + 0.3 * total / count, rel=1e-5)


# ----------------------------------------------------------------------------- no leakage
@pytest.mark.parametrize("n", [1, 2, 3, 4])
def test_a_position_never_sees_beyond_its_input_token(n):
    """Logits at position p may depend on tokens up to p + n (the input) and not on x(p + n + 1), the target."""
    model = _model(depth=4).eval()
    ids = _random_ids(1, seed=6)
    bb = model.backbone
    offsets = torch.tensor([n])

    def logits_for(tokens):
        with torch.no_grad():
            return model.mtp(bb.encode(tokens), tokens, None, bb.token_emb, bb.lm_head, offsets)

    base = logits_for(ids)
    for j in range(1, T):
        changed = ids.clone()
        changed[0, j] = (ids[0, j] % (VOCAB - 2)) + 1 if ids[0, j] != (ids[0, j] % (VOCAB - 2)) + 1 else 7
        assert changed[0, j] != ids[0, j]
        after = logits_for(changed)
        unaffected = max(j - n, 0)  # positions p with p + n < j
        assert torch.allclose(base[:, :unaffected], after[:, :unaffected], atol=1e-6), (n, j)
        if j - n >= 0:  # control: position j - n has x(j) as its INPUT and must notice the change
            assert (base[:, j - n] - after[:, j - n]).abs().max() > 1e-5, (n, j)


def test_changing_the_target_token_leaves_the_prediction_for_it_untouched():
    model = _model(depth=4).eval()
    ids = _random_ids(1, seed=7)
    bb, t, n = model.backbone, 3, 2
    other = ids.clone()
    other[0, t + n + 1] = (ids[0, t + n + 1] % (VOCAB - 2)) + 1  # the token position t has to predict
    with torch.no_grad():
        a = model.mtp(bb.encode(ids), ids, None, bb.token_emb, bb.lm_head, torch.tensor([n]))
        b = model.mtp(bb.encode(other), other, None, bb.token_emb, bb.lm_head, torch.tensor([n]))
    assert torch.allclose(a[:, : t + 1], b[:, : t + 1], atol=1e-6)


# ----------------------------------------------------------------------------- gradients
def test_the_mtp_loss_reaches_the_trunk_the_module_the_shared_head_and_the_offset_embedding():
    model = _model().train()
    ids = _random_ids(8, seed=8)
    batch = _batch(ids)
    _force_offsets(model, [1, 2, 3, 4])
    bb = model.backbone
    hidden = bb.encode(ids)
    loss, _ = offset_conditioned_mtp_loss(model, hidden, batch)  # the MTP part alone, no next-token loss
    loss.backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
    assert bb.layers[0].attn.heads[0].query_proj.weight.grad.abs().sum() > 0  # into the trunk
    assert bb.token_emb.weight.grad.abs().sum() > 0 and model.mtp.mtp_module.proj.weight.grad.abs().sum() > 0
    assert (model.mtp.offset_emb.weight.grad.abs().sum(1) > 0).all()  # every offset's embedding was used


def test_only_the_rows_of_the_offsets_that_were_used_get_a_gradient():
    model = _model().train()
    _force_offsets(model, [2])  # every sequence uses offset 2
    get_loss("autoregressive_mtp")(model, _batch(_random_ids(4, seed=9))).backward()
    row_gradient = model.mtp.offset_emb.weight.grad.abs().sum(1)
    assert row_gradient[1] > 0 and row_gradient[[0, 2, 3]].sum() == 0  # row n-1: the indexing is right


def test_the_trunk_gets_no_gradient_from_the_mtp_loss_when_its_hidden_states_are_detached():
    model = _model().train()
    ids = _random_ids(4, seed=10)
    hidden = model.backbone.encode(ids).detach()
    offset_conditioned_mtp_loss(model, hidden, _batch(ids))[0].backward()
    assert all(p.grad is None for p in model.backbone.layers.parameters())  # the control for the test above


# ----------------------------------------------------------------------------- masking
def test_targets_past_the_end_padding_and_other_documents_are_all_ignored():
    length = 10
    predictor = OffsetConditionedPredictor(_cfg(depth=4, max_length=length).model, positional=None)
    ids = _random_ids(1, length, seed=11)
    offsets = torch.tensor([2])
    # padding: the last two tokens are padding, so they are never targets
    mask = torch.ones(1, length, dtype=torch.long)
    mask[0, -2:] = 0
    padded = predictor.targets(_batch(ids, mask)["labels"], offsets)
    # real tokens are x_0 .. x_7; position t targets x(t + 3), so t = 0 .. 4 count and t >= 5 would target padding
    assert (padded[0, :5] != -100).all()
    assert (padded[0, 5:] == -100).all()
    # documents: [0 0 0 0 1 1 1 2 2 2]; the target must be in the position's own document
    documents = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 2, 2, 2]])
    with_documents = predictor.targets(_batch(ids)["labels"], offsets, documents)
    for t in range(length):
        in_range = t + 2 + 1 <= length - 1
        same_document = in_range and documents[0, t + 3] == documents[0, t]
        assert (with_documents[0, t] != -100) == bool(same_document), t
    plain = predictor.targets(_batch(ids)["labels"], offsets)  # without document ids only the end is masked
    assert (plain != -100).sum() == length - 3


def test_targets_past_the_end_are_ignored_even_when_the_last_label_is_a_real_token():
    """The end-of-sequence rule must not rely on the collate putting -100 in the last label slot."""
    length = 10
    predictor = OffsetConditionedPredictor(_cfg(depth=4, max_length=length).model, positional=None)
    ids = _random_ids(1, length, seed=15)
    labels = ids.clone()
    labels[:, :-1] = ids[:, 1:]  # the last label stays a real token (here: the last input token)
    for n in (1, 2, 3, 4):
        targets = predictor.targets(labels, torch.tensor([n]))
        assert (targets[0, : length - n - 1] != -100).all(), n  # t = 0 .. T-n-2 have a target ...
        assert (targets[0, length - n - 1 :] == -100).all(), n  # ... and the rest lie past the end


def test_a_padded_position_is_not_attended_to_by_later_positions():
    """The key-padding mask is handed through to the module's attention (checked with a mask in the middle)."""
    model = _model(depth=4).eval()
    predictor, bb = model.mtp, model.backbone
    torch.manual_seed(3)
    hidden = torch.randn(1, T, D)
    ids = _random_ids(1, seed=16)
    changed = hidden.clone()
    changed[0, 5] += 3.0  # perturb the module's input at position 5 (the trunk state there)
    offsets = torch.tensor([2])
    mask = torch.ones(1, T, dtype=torch.long)
    mask[0, 5] = 0  # position 5 is padding

    def logits(h, attention_mask):
        with torch.no_grad():
            return predictor(h, ids, attention_mask, bb.token_emb, bb.lm_head, offsets)

    assert torch.allclose(logits(hidden, mask)[:, 6:], logits(changed, mask)[:, 6:], atol=1e-6)  # masked: invisible later
    assert (logits(hidden, None)[:, 6:] - logits(changed, None)[:, 6:]).abs().max() > 1e-5  # control: visible when unmasked


def test_with_document_masking_the_mtp_never_mixes_documents():
    cfg = _cfg(depth=3)
    cfg.model.document_masking = True
    model = _model(cfg).eval()
    ids = _random_ids(1, seed=12)
    ids[0, 5] = 0  # an end-of-text token: tokens 0..5 are document 0, the rest document 1
    documents = torch.tensor([[0] * 6 + [1] * 8])
    other = ids.clone()
    other[0, 1:5] = (ids[0, 1:5] % (VOCAB - 2)) + 1  # rewrite most of document 0
    bb = model.backbone

    def mtp_logits(tokens, document_ids):
        with torch.no_grad():
            hidden = bb.encode(tokens, document_ids=document_ids)
            return model.mtp(hidden, tokens, None, bb.token_emb, bb.lm_head, torch.tensor([2]), document_ids)

    a, b = mtp_logits(ids, documents), mtp_logits(other, documents)
    assert torch.allclose(a[:, 6:], b[:, 6:], atol=1e-6)  # document 1 is unaffected by document 0
    leaky = mtp_logits(ids, None), mtp_logits(other, None)
    assert (leaky[0][:, 6:] - leaky[1][:, 6:]).abs().max() > 1e-5  # control: without the masks it leaks


# ----------------------------------------------------------------------------- the model, the loss and the config
def test_the_flag_is_off_by_default_and_selects_the_predictor():
    from hale_llm.config.sections.model import ModelConfig

    default = ModelConfig()
    assert (default.mtp_offset_conditioning, default.mtp_draft_tokens, default.mtp_offset_seed) == (False, 1, None)
    off, on = _model(_cfg(depth=3, offset=False)), _model(_cfg(depth=3, offset=True))
    assert isinstance(off.mtp, MultiTokenPredictor) and isinstance(on.mtp, OffsetConditionedPredictor)
    assert not any("offset_emb" in key or "sampler" in key for key in off.state_dict())
    assert not any(key.startswith("mtp.depths") for key in on.state_dict()) and any("offset_emb" in k for k in on.state_dict())


def test_one_module_replaces_the_chain_of_depths_so_it_is_smaller():
    depths = sum(p.numel() for p in _model(_cfg(depth=4, offset=False)).mtp.parameters())
    single = sum(p.numel() for p in _model(_cfg(depth=4, offset=True)).mtp.parameters())
    assert single < depths / 3  # one block set (+ a 4 x D table) instead of four


@pytest.mark.parametrize(
    "field,value,message",
    [("mtp_draft_tokens", 0, "mtp_draft_tokens"), ("mtp_draft_tokens", 5, "between 1 and mtp_depth=4"),
     ("mtp_offset_seed", -1, "mtp_offset_seed"), ("mtp_offset_seed", 2**31, "mtp_offset_seed")],
)
def test_invalid_settings_are_rejected(field, value, message):
    from hale_llm.config.sections.model import ModelConfig

    with pytest.raises(ValueError, match=message):
        ModelConfig(mtp_depth=4, **{field: value})


def test_the_context_window_must_fit_the_largest_offset():
    with pytest.raises(ValueError, match="context window of at least 6"):
        _model(_cfg(depth=4, max_length=5))


def test_the_loss_is_next_token_plus_lambda_times_the_sampled_offset_loss_and_eval_is_next_token_only():
    model = _model(weight=0.4).train()
    batch = _batch(_random_ids(4, seed=13))
    loss = get_loss("autoregressive_mtp")(model, batch)
    parts = model.loss_parts
    assert float(loss) == pytest.approx(float(parts["ntp_loss"]) + 0.4 * float(parts["mtp_loss"]), rel=1e-5)
    assert set(parts) == {"ntp_loss", "mtp_loss", "mtp_weight", "mtp_offset_mean"}
    cursor = model.mtp.sampler.cursor
    model.eval()
    eval_loss = get_loss("autoregressive_mtp")(model, batch)
    assert float(eval_loss) == pytest.approx(float(parts["ntp_loss"]), rel=1e-5)  # next-token only
    assert model.mtp.sampler.cursor == cursor  # evaluation draws no offsets


def test_every_training_loss_call_draws_one_offset_per_sequence():
    model = _model().train()
    for step in range(1, 4):
        get_loss("autoregressive_mtp")(model, _batch(_random_ids(5, seed=step)))
        assert model.mtp.sampler.cursor == 5 * step


def test_a_fresh_model_scores_about_ln_vocab_times_one_plus_lambda():
    import math

    model = _model(weight=0.3).train()
    losses = [float(get_loss("autoregressive_mtp")(model, _batch(_random_ids(16, seed=s)))) for s in range(4)]
    assert sum(losses) / 4 == pytest.approx(math.log(VOCAB) * 1.3, rel=0.05)


@pytest.mark.parametrize("ffn_type", ["mlp", "moe"])
def test_it_works_with_a_dense_and_a_mixture_of_experts_model(ffn_type):
    variant = "autoregressive_mtp" if ffn_type == "mlp" else "autoregressive_moe_mtp"
    model = _model(_cfg(depth=3, ffn_type=ffn_type, variant=variant)).train()
    loss = get_loss(variant)(model, _batch(_random_ids(4, seed=14)))
    loss.backward()
    assert torch.isfinite(loss)


def test_with_a_single_offset_it_is_deepseek_depth_one_in_every_sequence():
    model = _model(depth=1, offset=True).train()
    drawn = model.mtp.sample_offsets(50, torch.device("cpu"))
    assert set(drawn.tolist()) == {1}


def test_the_offset_seed_defaults_to_the_training_seed_and_can_be_overridden():
    cfg = _cfg()
    cfg.train.seed = 5
    default = _model(cfg).mtp.sampler
    assert default.seed == 5
    cfg.model.mtp_offset_seed = 77
    assert _model(cfg).mtp.sampler.seed == 77
    assert not torch.equal(BalancedOffsetSampler(4, 5).sample(80), BalancedOffsetSampler(4, 77).sample(80))


def test_the_offset_embedding_is_not_treated_as_a_hidden_matrix_by_muon_dion_or_weight_decay():
    model = _model()
    hidden, others = split_parameters(model.named_parameters())
    assert not any(p is model.mtp.offset_emb.weight for p in hidden)
    assert any(p is model.mtp.offset_emb.weight for p in others)
    assert any(p is model.mtp.mtp_module.proj.weight for p in hidden)  # the module's own matrices still go to Muon / Dion
    adamw = build_optimizer("adamw", model, {"lr": 1e-3, "weight_decay": 0.1, "adam_beta1": 0.9, "adam_beta2": 0.95})
    decayed = {id(p) for g in adamw.param_groups if g["weight_decay"] > 0 for p in g["params"]}
    assert id(model.mtp.offset_emb.weight) not in decayed and id(model.mtp.mtp_module.proj.weight) in decayed


def test_the_stream_position_survives_a_state_dict_round_trip_through_the_model():
    original = _model(seed=3).train()
    original.mtp.sample_offsets(23, torch.device("cpu"))
    clone = _model(seed=3)
    clone.load_state_dict(original.state_dict())
    assert clone.mtp.sampler.cursor == 23
    assert clone.mtp.sample_offsets(30, torch.device("cpu")).tolist() == original.mtp.sample_offsets(30, torch.device("cpu")).tolist()


def test_every_offset_gets_trained_on_a_counting_task_with_one_shared_module():
    """x(i) = i mod 20: every offset n has the answer x(t+n) + 1. After training, all offsets must be accurate."""
    cfg = _cfg(depth=4, weight=1.0)
    model = _model(cfg, seed=1).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
    for step in range(220):
        start = torch.randint(0, 20, (16, 1), generator=torch.Generator().manual_seed(step))
        ids = (start + torch.arange(T)) % 20 + 1
        loss = get_loss("autoregressive_mtp")(model, _batch(ids))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    model.eval()
    ids = (torch.randint(0, 20, (64, 1), generator=torch.Generator().manual_seed(999)) + torch.arange(T)) % 20 + 1
    bb = model.backbone
    with torch.no_grad():
        hidden = bb.encode(ids)
        for n in (1, 2, 3, 4):
            offsets = torch.full((64,), n)
            logits = model.mtp(hidden, ids, None, bb.token_emb, bb.lm_head, offsets)
            targets = model.mtp.targets(_batch(ids)["labels"], offsets)
            valid = targets != -100
            accuracy = (logits.argmax(-1)[valid] == targets[valid]).float().mean().item()
            assert accuracy > 0.9, (n, accuracy)


# ----------------------------------------------------------------------------- speculative decoding
def _plain_greedy(model, prompt, new_tokens):
    tokens = prompt.clone()
    with torch.no_grad():
        for _ in range(new_tokens):
            tokens = torch.cat([tokens, model.backbone(tokens)[:, -1].argmax(-1, keepdim=True)], dim=1)
    return tokens


@pytest.mark.parametrize("offset", [True, False])
@pytest.mark.parametrize("draft", [1, 2, 3, 4])
def test_greedy_speculative_decoding_equals_plain_greedy_decoding(offset, draft):
    model = _model(_cfg(depth=4, offset=offset, draft=draft, max_length=64), seed=4).eval()
    for seed in range(3):
        prompt = _random_ids(1, 6, seed=seed)
        decoder = SpeculativeDecoder(model, NO_EOS, temperature=0.0)
        assert torch.equal(decoder.generate(prompt, 30), _plain_greedy(model, prompt, 30)), (offset, draft, seed)


@pytest.mark.parametrize("offset", [True, False])
@pytest.mark.parametrize("draft", [1, 2, 3])
def test_each_round_drafts_exactly_mtp_draft_tokens(offset, draft):
    model = _model(_cfg(depth=4, offset=offset, draft=draft, max_length=64), seed=5).eval()
    decoder = SpeculativeDecoder(model, NO_EOS, temperature=0.0)
    decoder.generate(_random_ids(1, 6, seed=1), 25)
    assert decoder.rounds > 0 and decoder.drafts_proposed == decoder.rounds * draft
    assert len(decoder.depth_reached) == draft


def test_the_default_is_one_draft_per_round_in_both_modes():
    for offset in (True, False):
        model = _model(_cfg(depth=3, offset=offset, draft=1, max_length=64), seed=6).eval()
        decoder = SpeculativeDecoder(model, NO_EOS, temperature=0.0)
        decoder.generate(_random_ids(1, 6, seed=2), 20)
        assert decoder.drafts_proposed == decoder.rounds


def test_sampling_with_a_temperature_runs_and_respects_the_length():
    model = _model(_cfg(depth=4, draft=3, max_length=64), seed=7).eval()
    torch.manual_seed(0)
    output = SpeculativeDecoder(model, NO_EOS, temperature=1.0).generate(_random_ids(1, 6, seed=3), 25)
    assert output.shape == (1, 6 + 25) and 0 <= int(output.min()) and int(output.max()) < VOCAB


def test_the_decoding_step_is_the_same_computation_as_the_training_forward():
    """step(k) over a prefix must equal forward() with offset k, read at the last prefix position."""
    model = _model(_cfg(depth=4, max_length=32), seed=8).eval()
    bb, mtp = model.backbone, model.mtp
    tokens = _random_ids(1, 12, seed=4)
    L = 9  # committed prefix; `known` must hold x_0 .. x_{L + k - 1}
    with torch.no_grad():
        hidden = bb.encode(tokens[:, :L])
        for k in (1, 2, 3):
            _, last = mtp.step(k, hidden, tokens[:, : L + k], bb.token_emb, bb.lm_head)
            full_hidden = torch.cat([hidden, torch.zeros(1, k, D)], dim=1)  # positions after L - 1 cannot matter (causal)
            logits = mtp(full_hidden, tokens[:, : L + k], None, bb.token_emb, bb.lm_head, torch.tensor([k]))
            assert torch.allclose(last, logits[:, L - 1], atol=1e-5), k


def test_offset_drafts_all_read_the_trunk_states_while_deepseek_depths_read_each_other():
    calls = {}
    for offset in (True, False):
        model = _model(_cfg(depth=3, offset=offset, draft=3, max_length=64), seed=9).eval()
        seen, real_step = [], model.mtp.step

        def spy(k, previous, tokens, token_emb, lm_head, _real=real_step, _seen=seen):
            _seen.append(previous)
            return _real(k, previous, tokens, token_emb, lm_head)

        model.mtp.step = spy
        SpeculativeDecoder(model, NO_EOS, 0.0).generate(_random_ids(1, 6, seed=5), 4)
        calls[offset] = seen[:3]  # the three drafts of the first round
    assert calls[True][0] is calls[True][1] is calls[True][2]  # the same trunk hidden states every time
    assert calls[False][0] is not calls[False][1] and calls[False][1] is not calls[False][2]  # chained


# ----------------------------------------------------------------------------- training and resuming
class _Stop(Exception):
    pass


def _interrupt_after(step_count):
    def hook(step):
        if step + 1 == step_count:
            raise _Stop()
    return hook


def _train_cfg(tmp_path, steps=40, batch_size=3):
    cfg = load_config("configs/autoregressive_mtp/dense_nano.yaml")
    cfg.train.steps, cfg.train.epochs, cfg.train.batch_size = steps, None, batch_size
    cfg.train.resume, cfg.train.log_every = True, 100
    cfg.train.checkpoint_path = str(tmp_path / "run" / "model.pt")
    cfg.train.lean.save_every, cfg.train.lean.eval_every, cfg.train.lean.keep_best = 10, 0, 0
    cfg.train.lean.warmup_steps, cfg.train.lean.compile, cfg.train.lean.precision = 2, False, "fp32"
    m = cfg.model
    m.d_model, m.n_heads, m.n_layers, m.d_ff, m.dropout = 32, 4, 1, 64, 0.0
    m.mtp_depth, m.mtp_offset_conditioning, m.mtp_draft_tokens = 4, True, 2
    cfg.data.source, cfg.data.overfit_text, cfg.data.n_overfit_copies = "overfit", "the cat sat on the mat and the dog sat on the log", 12
    cfg.set_context_window(16)
    return cfg


def test_training_uses_every_offset_equally_often(tiny_tokenizer, tmp_path):
    from hale_llm.training.lean_trainer import train_lean

    cfg = _train_cfg(tmp_path, steps=12, batch_size=3)  # 36 sequences = 9 decks of 4 offsets
    cfg.train.lean.save_every = 0
    train_lean(cfg, tokenizer=tiny_tokenizer)
    state = torch.load(cfg.train.checkpoint_path, weights_only=False)["model_state_dict"]["mtp.sampler._extra_state"]
    assert state["cursor"] == 36 and state["n_offsets"] == 4
    drawn = BalancedOffsetSampler(4, state["seed"]).sample(state["cursor"])
    assert [int((drawn == n).sum()) for n in (1, 2, 3, 4)] == [9, 9, 9, 9]


def test_an_interrupted_run_resumes_to_exactly_the_uninterrupted_result_including_the_offsets(tiny_tokenizer, tmp_path):
    from hale_llm.training.lean_trainer import train_lean

    reference = train_lean(_train_cfg(tmp_path / "ref"), tokenizer=tiny_tokenizer)
    run = _train_cfg(tmp_path / "run")
    with pytest.raises(_Stop):
        train_lean(run, tokenizer=tiny_tokenizer, on_step=_interrupt_after(24))  # the last resume checkpoint is from step 20
    resumed = train_lean(run, tokenizer=tiny_tokenizer)
    assert len(resumed) == 20 and resumed == pytest.approx(reference[20:], abs=1e-6)
    final = lambda path: torch.load(path, weights_only=False)["model_state_dict"]  # noqa: E731
    a, b = final(_train_cfg(tmp_path / "ref").train.checkpoint_path), final(run.train.checkpoint_path)
    assert a["mtp.sampler._extra_state"] == b["mtp.sampler._extra_state"] and a["mtp.sampler._extra_state"]["cursor"] == 40 * 3
    assert all(torch.allclose(a[k], b[k], atol=1e-6) for k in a if not k.endswith("_extra_state"))  # float rounding only (~1e-8)


@pytest.mark.parametrize("optimizer", ["adamw", "muon", "dion"])
def test_it_trains_with_every_optimizer(optimizer, tiny_tokenizer, tmp_path):
    from hale_llm.training.lean_trainer import train_lean

    cfg = _train_cfg(tmp_path, steps=30)
    cfg.train.optimizer_type, cfg.train.lean.save_every = optimizer, 0
    losses = train_lean(cfg, tokenizer=tiny_tokenizer)
    assert all(map(torch.isfinite, map(torch.tensor, losses))) and losses[-1] < losses[0]
