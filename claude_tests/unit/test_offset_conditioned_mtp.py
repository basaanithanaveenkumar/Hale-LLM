"""Offset-conditioned MTP: stratified offsets over the global batch, the alignment, V3 depth-1 equivalence, the summed
stratified loss, leakage, masking, gradients, resuming and speculative decoding."""

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
from hale_llm.nn.mtp import MultiTokenPredictor, OffsetConditionedPredictor, OffsetShare, StratifiedOffsetAssigner
from hale_llm.optim import build_optimizer
from hale_llm.optim.parameter_groups import split_parameters
from hale_llm.registry import get_loss, get_variant

VOCAB, T, D = 40, 14, 32
NO_EOS = type("Tok", (), {"eos_token_id": None})()


# ----------------------------------------------------------------------------- helpers
def _cfg(depth=4, offset=True, draft=1, weight=0.3, ffn_type="mlp", n_layers=1, max_length=T, variant="autoregressive_mtp",
         reduction=None, accum=1, remainder="balanced"):
    cfg = load_config("configs/autoregressive_mtp/dense_nano.yaml")
    cfg.variant = variant
    cfg.set_context_window(max_length)
    m = cfg.model
    m.d_model, m.n_heads, m.n_layers, m.d_ff, m.dropout = D, 4, 2, 64, 0.0
    m.ffn_type, m.moe_hidden_dim, m.moe_router = ffn_type, 48, "topk"
    m.mtp_depth, m.mtp_offset_conditioning, m.mtp_draft_tokens = depth, offset, draft
    m.mtp_loss_weight, m.mtp_n_layers = weight, n_layers
    m.mtp_loss_reduction, m.mtp_offset_remainder = reduction, remainder
    cfg.train.lean.accum_steps = accum
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
    """Make the assigner hand out ``values`` in a cycle, each call treated as a whole step (no plan, no accumulation)."""
    cycle = itertools.cycle(values)
    n = model.mtp.depth

    def next_share(batch_size, device="cpu"):
        offsets = torch.tensor([next(cycle) for _ in range(batch_size)], device=device)
        return OffsetShare(offsets, torch.bincount(offsets - 1, minlength=n), 1)

    model.mtp.assigner.next_share = next_share


class _Recorder(torch.nn.Module):
    """A stand-in for the token embedding: remembers the ids it was asked about and returns zeros."""

    def __init__(self):
        super().__init__()
        self.seen = None

    def forward(self, ids):
        self.seen = ids.clone()
        return torch.zeros(*ids.shape, D)


# ----------------------------------------------------------------------------- the assigner (stratified offsets)
def _assigner(n=4, seed=3, accum=1, remainder="balanced"):
    return StratifiedOffsetAssigner(n, seed, accum, remainder)


def _counts(offsets, n):
    tally = collections.Counter(offsets.tolist())
    return [tally[k] for k in range(1, n + 1)]


def test_offsets_are_integers_in_range():
    share = _assigner().next_share(1000)
    assert share.offsets.dtype == torch.long and share.offsets.shape == (1000,)
    assert 1 <= int(share.offsets.min()) and int(share.offsets.max()) <= 4


@pytest.mark.parametrize("n_offsets", [1, 2, 3, 4, 5, 6])
@pytest.mark.parametrize("batch", list(range(1, 27)))
def test_one_step_is_stratified_floor_of_g_over_n_each_and_the_remainder_distinct(n_offsets, batch):
    share = _assigner(n_offsets).next_share(batch)
    counts = _counts(share.offsets, n_offsets)
    base, rest = divmod(batch, n_offsets)
    assert sum(counts) == batch and all(count in (base, base + 1) for count in counts), counts
    if rest:
        assert sum(count == base + 1 for count in counts) == rest  # exactly G mod N offsets got the extra sequence
    else:
        assert len(set(counts)) == 1  # a multiple of N: all equal
    assert share.global_counts.tolist() == counts  # the share reports the step's counts


def test_a_multiple_of_n_gives_every_offset_exactly_g_over_n_times_in_every_step():
    assigner = _assigner(4, seed=9)
    for _ in range(60):
        assert _counts(assigner.next_share(12).offsets, 4) == [3, 3, 3, 3]


def test_the_order_is_shuffled_not_a_repeated_1_to_n_pattern():
    plans = [tuple(_assigner(4, seed=seed).next_share(8).offsets.tolist()) for seed in range(300)]
    assert len(set(plans)) > 100  # many different orders of the same multiset
    first = collections.Counter(plan.index(1) for plan in plans)  # where offset 1 first appears among the 8 slots
    assert set(first) <= set(range(7))  # two copies of 1 in 8 slots: the first one is at position 0..6
    assert all(first[position] > 20 for position in range(4))  # and the likely positions all occur (expected 75..43 of 300)
    assert all(sorted(plan) == [1, 1, 2, 2, 3, 3, 4, 4] for plan in plans)


def test_same_seed_same_plan_different_seed_or_step_different_plan():
    a, b, other = _assigner(seed=1), _assigner(seed=1), _assigner(seed=2)
    steps = [[x.next_share(16).offsets.tolist() for _ in range(6)] for x in (a, b, other)]
    assert steps[0] == steps[1] and steps[0] != steps[2]
    assert len({tuple(step) for step in steps[0]}) > 1  # a new shuffle every step


@pytest.mark.parametrize("batch,accum", [(1, 4), (2, 2), (4, 1), (3, 2), (6, 1), (2, 3), (5, 4)])
def test_the_split_into_micro_batches_does_not_change_the_plan(batch, accum):
    """The same global batch is planned the same way whether it is one big batch or several accumulated micro-batches."""
    split, whole = _assigner(seed=5, accum=accum), _assigner(seed=5, accum=1)
    for _ in range(5):
        shares = [split.next_share(batch) for _ in range(accum)]
        assert torch.equal(torch.cat([share.offsets for share in shares]), whole.next_share(batch * accum).offsets)
        assert all(share.micro_batches == accum for share in shares)
        assert all(torch.equal(share.global_counts, shares[0].global_counts) for share in shares)


def test_a_micro_batch_of_one_has_one_offset_and_the_accumulated_step_sees_them_all():
    assigner = _assigner(4, seed=7, accum=4)
    for _ in range(25):
        shares = [assigner.next_share(1) for _ in range(4)]
        assert all(share.offsets.numel() == 1 for share in shares)
        assert sorted(int(share.offsets) for share in shares) == [1, 2, 3, 4]  # every offset exactly once per step
        assert shares[0].global_counts.tolist() == [1, 1, 1, 1] and shares[0].micro_batches == 4


@pytest.mark.parametrize("world,accum,batch", [(2, 2, 3), (2, 1, 4), (4, 1, 2), (3, 2, 2), (2, 3, 1)])
def test_every_rank_gets_its_share_of_one_global_plan_and_no_rank_draws_on_its_own(world, accum, batch):
    ranks = [_assigner(4, seed=11, accum=accum) for _ in range(world)]
    for rank, assigner in enumerate(ranks):
        assigner.set_data_parallel(rank, world)
    reference = _assigner(4, seed=11, accum=1)
    for _ in range(6):
        taken = {}  # (micro, rank) -> the offsets that rank received for that micro-batch
        for micro in range(accum):
            for rank, assigner in enumerate(ranks):
                share = assigner.next_share(batch)
                taken[micro, rank] = share.offsets
                assert share.micro_batches == accum * world
        # The shares, in the documented order (micro-major, then rank), are exactly the single-process plan.
        global_batch = torch.cat([taken[micro, rank] for micro in range(accum) for rank in range(world)])
        assert torch.equal(global_batch, reference.next_share(batch * accum * world).offsets)
        total = accum * world * batch
        if total % 4 == 0:
            assert _counts(global_batch, 4) == [total // 4] * 4  # a multiple of N: exactly equal


def test_data_parallel_arguments_are_validated():
    assigner = _assigner()
    for rank, world in [(0, 0), (2, 2), (-1, 2)]:
        with pytest.raises(ValueError, match="rank"):
            assigner.set_data_parallel(rank, world)


def test_balanced_remainder_keeps_every_total_within_one_and_each_step_distinct():
    assigner = _assigner(4, seed=2, remainder="balanced")
    totals = collections.Counter()
    for step in range(120):
        offsets = assigner.next_share(3).offsets.tolist()  # G = 3 of N = 4: all three are remainder
        assert len(set(offsets)) == 3  # distinct offsets
        totals.update(offsets)
        spread = [totals[k] for k in (1, 2, 3, 4)]
        assert max(spread) - min(spread) <= 1, (step, spread)
    assert [totals[k] for k in (1, 2, 3, 4)] == [90] * 4  # 120 steps x 3 = 360 = 4 x 90


def test_balanced_remainder_with_a_partial_round_on_top_of_full_rounds():
    assigner = _assigner(4, seed=4, remainder="balanced")
    totals = collections.Counter()
    for _ in range(100):  # G = 6 = one full round + 2 remainder offsets
        totals.update(assigner.next_share(6).offsets.tolist())
    assert [totals[k] for k in (1, 2, 3, 4)] == [150] * 4  # 100 x (1 + 2/4) each


def test_random_remainder_is_distinct_within_a_step_but_has_no_memory_across_steps():
    assigner = _assigner(4, seed=2, remainder="random")
    totals, worst = collections.Counter(), 0
    for _ in range(300):
        offsets = assigner.next_share(3).offsets.tolist()
        assert len(set(offsets)) == 3
        totals.update(offsets)
        spread = [totals[k] for k in (1, 2, 3, 4)]
        worst = max(worst, max(spread) - min(spread))
    assert worst >= 2  # unlike "balanced", the running totals drift apart ...
    assert all(0.8 * 225 < totals[k] < 1.2 * 225 for k in (1, 2, 3, 4)), totals  # ... but stay equal in expectation


def test_the_remainder_offset_is_random_not_always_the_same():
    chosen = {int(_assigner(4, seed=seed).next_share(1).offsets[0]) for seed in range(200)}
    assert chosen == {1, 2, 3, 4}


def test_planning_does_not_touch_the_global_random_state():
    state = torch.get_rng_state()
    assigner = _assigner()
    for _ in range(20):
        assigner.next_share(7)
    assert torch.equal(state, torch.get_rng_state())


def test_the_position_is_saved_and_restored_also_in_the_middle_of_a_step():
    for consumed in (0, 1, 2):  # micro-batches already taken in the current step (accum = 3)
        original = _assigner(4, seed=6, accum=3, remainder="balanced")
        for _ in range(7 + consumed):  # a few whole steps (3 micro-batches of 2 each), then part of one
            original.next_share(2)
        saved = copy_state(original)
        expected = [original.next_share(2).offsets.tolist() for _ in range(15)]
        restored = _assigner(4, seed=6, accum=3, remainder="balanced")
        restored.set_extra_state(saved)
        assert [restored.next_share(2).offsets.tolist() for _ in range(15)] == expected, consumed


def copy_state(assigner):
    import copy

    return copy.deepcopy(assigner.get_extra_state())


def test_a_foreign_or_inconsistent_state_is_refused():
    base = _assigner(4, seed=6, accum=3)
    base.next_share(2)  # part-way through a step
    state = copy_state(base)
    for other in (_assigner(4, seed=7, accum=3), _assigner(3, seed=6, accum=3), _assigner(4, seed=6, accum=3, remainder="random")):
        with pytest.raises(ValueError, match="offset assigner state is for"):
            other.set_extra_state(state)
    with pytest.raises(ValueError, match="part-way through a step"):
        _assigner(4, seed=6, accum=2).set_extra_state(state)  # a different accum_steps mid-step
    boundary = _assigner(4, seed=6, accum=3)
    boundary.next_share(2), boundary.next_share(2), boundary.next_share(2)  # a whole step: position is a step boundary
    _assigner(4, seed=6, accum=2).set_extra_state(copy_state(boundary))  # fine at a boundary


def test_the_micro_batch_size_cannot_change_inside_a_step():
    assigner = _assigner(4, accum=2)
    assigner.next_share(3)
    with pytest.raises(ValueError, match="micro-batch size changed"):
        assigner.next_share(4)


def test_invalid_assigner_arguments_are_refused():
    for kwargs, message in [({"n_offsets": 0}, "at least 1"), ({"seed": -1}, "seed"), ({"seed": 2**31}, "seed"),
                            ({"accum_steps": 0}, "accum_steps"), ({"remainder": "x"}, "remainder")]:
        with pytest.raises(ValueError, match=message):
            StratifiedOffsetAssigner(**{"n_offsets": 4, **kwargs})
    with pytest.raises(ValueError, match="at least 1"):
        _assigner().next_share(0)


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


def _reference_offset_terms(model, ids, offsets):
    """For each offset n: the mean over its sequences of l_s, where l_s is the mean CE over the sequence's valid positions.

    Written with plain loops and raw token ids as targets, independent of ``targets()`` and of the weighting in the loss.
    """
    bb = model.backbone
    with torch.no_grad():
        hidden = bb.encode(ids)
        logits = model.mtp(hidden, ids, None, bb.token_emb, bb.lm_head, torch.tensor(offsets))
    per_offset = collections.defaultdict(list)
    for b, n in enumerate(offsets):
        losses = [float(F.cross_entropy(logits[b, t][None], ids[b, t + n + 1][None]))
                  for t in range(T) if t + n + 1 <= T - 1]  # the module reads x(t+n) and predicts x(t+n+1)
        per_offset[n].append(sum(losses) / len(losses))
    return {n: sum(values) / len(values) for n, values in per_offset.items()}


def test_the_mtp_term_is_the_sum_over_offsets_of_each_offsets_mean_loss():
    """L_1 + L_2 + L_3 + L_4, each L_n averaged over its own sequences: added up, not averaged over the offsets."""
    model = _model(weight=0.3).train()
    ids = _random_ids(6, seed=5)
    offsets = [1, 2, 3, 4, 2, 1]
    _force_offsets(model, offsets)
    per_offset = _reference_offset_terms(model, ids, offsets)
    bb = model.backbone
    with torch.no_grad():
        hidden = bb.encode(ids)
        ntp = F.cross_entropy(bb.lm_head(hidden)[:, :-1].reshape(-1, VOCAB), ids[:, 1:].reshape(-1))
        loss = get_loss("autoregressive_mtp")(model, _batch(ids))
    expected_sum = sum(per_offset.values())
    assert float(model.loss_parts["mtp_loss"]) == pytest.approx(expected_sum, rel=1e-5)
    assert float(model.loss_parts["mtp_loss"]) > 3.0 * (expected_sum / 4)  # four terms were added, not averaged
    assert float(loss) == pytest.approx(float(ntp) + 0.3 * expected_sum, rel=1e-5)


def test_each_offset_counts_the_same_however_many_sequences_it_has_stratified_not_token_averaged():
    """Offset 1 on three sequences and offset 2 on one: the term is mean(offset 1) + (offset 2), not a mean of four."""
    model = _model().train()
    ids = _random_ids(4, seed=6)
    offsets = [1, 1, 1, 2]
    _force_offsets(model, offsets)
    per_offset = _reference_offset_terms(model, ids, offsets)
    get_loss("autoregressive_mtp")(model, _batch(ids))
    assert float(model.loss_parts["mtp_loss"]) == pytest.approx(per_offset[1] + per_offset[2], rel=1e-5)


def test_mean_reduction_divides_the_sum_by_the_number_of_offsets():
    ids, offsets = _random_ids(8, seed=7), [1, 2, 3, 4, 4, 3, 2, 1]
    summed, averaged = _model(reduction="sum", seed=3).train(), _model(reduction="mean", seed=3).train()
    for model in (summed, averaged):
        _force_offsets(model, offsets)
        get_loss("autoregressive_mtp")(model, _batch(ids))
    assert float(averaged.loss_parts["mtp_loss"]) == pytest.approx(float(summed.loss_parts["mtp_loss"]) / 4, rel=1e-5)


def test_the_default_reduction_is_a_sum_for_offsets_and_a_mean_for_deepseek_depths():
    from hale_llm.config.sections.model import ModelConfig

    assert ModelConfig(mtp_offset_conditioning=True, mtp_depth=4).mtp_reduction() == "sum"
    assert ModelConfig(mtp_offset_conditioning=False, mtp_depth=4).mtp_reduction() == "mean"
    assert ModelConfig(mtp_offset_conditioning=False, mtp_depth=4, mtp_loss_reduction="sum").mtp_reduction() == "sum"
    assert ModelConfig(mtp_offset_conditioning=True, mtp_depth=4, mtp_loss_reduction="mean").mtp_loss_terms() == 1
    assert ModelConfig(mtp_offset_conditioning=True, mtp_depth=4).mtp_loss_terms() == 4


def test_deepseek_depths_can_also_be_added_up_instead_of_averaged():
    mean_model, sum_model = _model(_cfg(depth=3, offset=False), seed=4).train(), _model(_cfg(depth=3, offset=False, reduction="sum"), seed=4).train()
    batch = _batch(_random_ids(4, seed=8))
    get_loss("autoregressive_mtp")(mean_model, batch)
    get_loss("autoregressive_mtp")(sum_model, batch)
    depth_sum = sum(float(sum_model.loss_parts[f"mtp_loss_depth{k}"]) for k in (1, 2, 3))
    assert float(sum_model.loss_parts["mtp_loss"]) == pytest.approx(depth_sum, rel=1e-5)
    assert float(mean_model.loss_parts["mtp_loss"]) == pytest.approx(depth_sum / 3, rel=1e-5)  # the paper's (1/D) sum


def test_the_initial_loss_scale_counts_the_summed_terms():
    cls = get_variant("autoregressive_mtp")
    assert cls.initial_loss_scale(_cfg(depth=4, weight=0.3)) == pytest.approx(1 + 0.3 * 4)  # offsets: summed by default
    assert cls.initial_loss_scale(_cfg(depth=4, weight=0.3, reduction="mean")) == pytest.approx(1.3)
    assert cls.initial_loss_scale(_cfg(depth=3, offset=False, weight=0.3)) == pytest.approx(1.3)  # DeepSeek: the paper's mean
    assert cls.initial_loss_scale(_cfg(depth=3, offset=False, weight=0.3, reduction="sum")) == pytest.approx(1 + 0.3 * 3)


# ----------------------------------------------------------------------------- accumulation and several ranks
def _grads(model):
    return {name: p.grad.clone() for name, p in model.named_parameters() if p.grad is not None}


def _fresh_pair(accum_a, accum_b, seed=0):
    """Two models with identical weights, each with its assigner at step 0 (load_state_dict also copies the position)."""
    a, b = _model(_cfg(accum=accum_a), seed=seed).train(), _model(_cfg(accum=accum_b), seed=seed).train()
    b.load_state_dict(a.state_dict())
    for model in (a, b):
        model.mtp.assigner.step = model.mtp.assigner.micro = 0
        model.mtp.assigner._plan = None
        model.mtp.assigner._extra_counts = [0] * model.mtp.depth
    return a, b


@pytest.mark.parametrize("batch,accum", [(1, 4), (2, 2)])
def test_accumulated_micro_batches_give_the_same_loss_and_gradients_as_one_big_batch(batch, accum):
    """What the trainer does: divide each micro-batch loss by accum and add the gradients up."""
    whole, split = _fresh_pair(1, accum)
    ids = _random_ids(4, seed=21)
    loss_whole = get_loss("autoregressive_mtp")(whole, _batch(ids))
    loss_whole.backward()
    total, mtp_total = 0.0, 0.0
    for m in range(accum):
        micro = get_loss("autoregressive_mtp")(split, _batch(ids[m * batch : (m + 1) * batch])) / accum
        micro.backward()
        total += float(micro)
        mtp_total += float(split.loss_parts["mtp_loss"]) / accum
    assert total == pytest.approx(float(loss_whole), abs=1e-5)
    assert mtp_total == pytest.approx(float(whole.loss_parts["mtp_loss"]), abs=1e-5)  # still the sum over offsets
    g_whole, g_split = _grads(whole), _grads(split)
    assert g_whole.keys() == g_split.keys() and all(torch.allclose(g_whole[k], g_split[k], atol=1e-5) for k in g_whole)


def test_a_micro_batch_of_one_still_adds_the_offsets_up():
    """Each micro-batch has a single offset, yet the accumulated step carries L_1 + ... + L_4, not their mean."""
    whole, split = _fresh_pair(1, 4, seed=1)
    ids = _random_ids(4, seed=22)
    get_loss("autoregressive_mtp")(whole, _batch(ids))
    per_micro = []
    for m in range(4):
        get_loss("autoregressive_mtp")(split, _batch(ids[m : m + 1]))
        per_micro.append(float(split.loss_parts["mtp_loss"]))
    summed = float(whole.loss_parts["mtp_loss"])
    assert sum(per_micro) / 4 == pytest.approx(summed, abs=1e-5)  # what the trainer logs: the average of the micro-batches
    assert min(per_micro) > 0 and all(value > summed / 4 for value in per_micro)  # each micro-batch is scaled up by N


def test_two_data_parallel_ranks_together_equal_one_big_batch():
    """2 ranks x 2 micro-batches x 1 sequence: the offsets come from one global plan and the averaged gradients match."""
    whole = _model(_cfg(accum=1), seed=2).train()
    ranks = []
    for rank in range(2):
        model = _model(_cfg(accum=2), seed=2).train()
        model.load_state_dict(whole.state_dict())
        model.mtp.assigner.step = model.mtp.assigner.micro = 0
        model.mtp.assigner._plan = None
        model.mtp.assigner.set_data_parallel(rank, 2)
        ranks.append(model)
    whole.mtp.assigner.step = whole.mtp.assigner.micro = 0
    whole.mtp.assigner._plan = None
    ids = _random_ids(4, seed=23)
    loss_whole = get_loss("autoregressive_mtp")(whole, _batch(ids))
    loss_whole.backward()
    losses = []
    for rank, model in enumerate(ranks):
        total = 0.0
        for micro in range(2):
            index = micro * 2 + rank  # the documented order: micro-batch major, then rank
            loss = get_loss("autoregressive_mtp")(model, _batch(ids[index : index + 1])) / 2
            loss.backward()
            total += float(loss)
        losses.append(total)
    assert sum(losses) / 2 == pytest.approx(float(loss_whole), abs=1e-5)  # data-parallel averages the ranks
    for name, parameter in whole.named_parameters():
        if parameter.grad is None:
            continue
        mean_grad = (dict(ranks[0].named_parameters())[name].grad + dict(ranks[1].named_parameters())[name].grad) / 2
        assert torch.allclose(parameter.grad, mean_grad, atol=1e-5), name


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
    assert not any("offset_emb" in key or "assigner" in key for key in off.state_dict())
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
    position = (model.mtp.assigner.step, model.mtp.assigner.micro)
    model.eval()
    eval_loss = get_loss("autoregressive_mtp")(model, batch)
    assert float(eval_loss) == pytest.approx(float(parts["ntp_loss"]), rel=1e-5)  # next-token only
    assert (model.mtp.assigner.step, model.mtp.assigner.micro) == position  # evaluation plans no offsets


def test_every_training_loss_call_takes_one_micro_batch_and_a_step_ends_after_accum_of_them():
    model = _model(_cfg(accum=3)).train()
    for call in range(1, 10):
        get_loss("autoregressive_mtp")(model, _batch(_random_ids(5, seed=call)))
        assert (model.mtp.assigner.step, model.mtp.assigner.micro) == divmod(call, 3)


def test_a_fresh_model_scores_about_ln_vocab_times_the_initial_loss_scale():
    import math

    cfg = _cfg(weight=0.3)
    model = _model(cfg).train()
    losses = [float(get_loss("autoregressive_mtp")(model, _batch(_random_ids(16, seed=s)))) for s in range(4)]
    scale = get_variant("autoregressive_mtp").initial_loss_scale(cfg)  # 1 + 0.3 * 4 offsets summed
    assert sum(losses) / 4 == pytest.approx(math.log(VOCAB) * scale, rel=0.05)


@pytest.mark.parametrize("ffn_type", ["mlp", "moe"])
def test_it_works_with_a_dense_and_a_mixture_of_experts_model(ffn_type):
    variant = "autoregressive_mtp" if ffn_type == "mlp" else "autoregressive_moe_mtp"
    model = _model(_cfg(depth=3, ffn_type=ffn_type, variant=variant)).train()
    loss = get_loss(variant)(model, _batch(_random_ids(4, seed=14)))
    loss.backward()
    assert torch.isfinite(loss)


def test_with_a_single_offset_it_is_deepseek_depth_one_in_every_sequence():
    model = _model(depth=1, offset=True).train()
    assert set(model.mtp.next_offsets(50, torch.device("cpu")).offsets.tolist()) == {1}


def test_the_offset_seed_defaults_to_the_training_seed_and_can_be_overridden():
    cfg = _cfg()
    cfg.train.seed = 5
    default = _model(cfg).mtp.assigner
    assert default.seed == 5
    cfg.model.mtp_offset_seed = 77
    overridden = _model(cfg).mtp.assigner
    assert overridden.seed == 77
    assert not torch.equal(default.next_share(40).offsets, overridden.next_share(40).offsets)


def test_the_offset_embedding_is_not_treated_as_a_hidden_matrix_by_muon_dion_or_weight_decay():
    model = _model()
    hidden, others = split_parameters(model.named_parameters())
    assert not any(p is model.mtp.offset_emb.weight for p in hidden)
    assert any(p is model.mtp.offset_emb.weight for p in others)
    assert any(p is model.mtp.mtp_module.proj.weight for p in hidden)  # the module's own matrices still go to Muon / Dion
    adamw = build_optimizer("adamw", model, {"lr": 1e-3, "weight_decay": 0.1, "adam_beta1": 0.9, "adam_beta2": 0.95})
    decayed = {id(p) for g in adamw.param_groups if g["weight_decay"] > 0 for p in g["params"]}
    assert id(model.mtp.offset_emb.weight) not in decayed and id(model.mtp.mtp_module.proj.weight) in decayed


def test_the_assigner_position_survives_a_state_dict_round_trip_through_the_model():
    original = _model(_cfg(accum=2), seed=3).train()
    for _ in range(5):
        original.mtp.next_offsets(3, torch.device("cpu"))  # 2 whole steps and one micro-batch of the third
    clone = _model(_cfg(accum=2), seed=3)
    clone.load_state_dict(original.state_dict())
    assert (clone.mtp.assigner.step, clone.mtp.assigner.micro) == (2, 1)
    expected = [original.mtp.next_offsets(3, torch.device("cpu")).offsets.tolist() for _ in range(6)]
    assert [clone.mtp.next_offsets(3, torch.device("cpu")).offsets.tolist() for _ in range(6)] == expected


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


def _train_cfg(tmp_path, steps=40, batch_size=3, accum=1):
    cfg = load_config("configs/autoregressive_mtp/dense_nano.yaml")
    cfg.train.steps, cfg.train.epochs, cfg.train.batch_size = steps, None, batch_size
    cfg.train.lean.accum_steps = accum
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


@pytest.mark.parametrize("batch,accum,extras", [(3, 1, [9, 9, 9, 9]), (1, 4, [0, 0, 0, 0]), (2, 3, [6, 6, 6, 6]), (4, 1, [0, 0, 0, 0])])
def test_training_stratifies_the_global_batch_and_every_offset_is_used_equally_often(batch, accum, extras, tiny_tokenizer, tmp_path):
    from hale_llm.training.lean_trainer import train_lean

    cfg = _train_cfg(tmp_path, steps=12, batch_size=batch, accum=accum)
    cfg.train.lean.save_every = 0
    train_lean(cfg, tokenizer=tiny_tokenizer)
    state = torch.load(cfg.train.checkpoint_path, weights_only=False)["model_state_dict"]["mtp.assigner._extra_state"]
    assert (state["step"], state["micro"], state["n_offsets"]) == (12, 0, 4)  # 12 whole steps, none left half-done
    # G = batch x accum sequences per step. The multiple-of-N part is exactly equal in every step; the remainder
    # offsets (G mod 4 of them per step) were handed out so that the totals over the 12 steps are equal too.
    assert state["extra_counts"] == extras


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
    key = "mtp.assigner._extra_state"
    assert a[key] == b[key] and a[key]["step"] == 40 and a[key]["extra_counts"] == [30, 30, 30, 30]  # 40 steps x 3 extras
    assert all(torch.allclose(a[k], b[k], atol=1e-6) for k in a if not k.endswith("_extra_state"))  # float rounding only (~1e-8)


def test_resuming_with_gradient_accumulation_also_reproduces_the_offsets(tiny_tokenizer, tmp_path):
    from hale_llm.training.lean_trainer import train_lean

    reference = train_lean(_train_cfg(tmp_path / "ref", accum=2, batch_size=2), tokenizer=tiny_tokenizer)
    run = _train_cfg(tmp_path / "run", accum=2, batch_size=2)
    with pytest.raises(_Stop):
        train_lean(run, tokenizer=tiny_tokenizer, on_step=_interrupt_after(24))
    resumed = train_lean(run, tokenizer=tiny_tokenizer)
    assert resumed == pytest.approx(reference[20:], abs=1e-6)


@pytest.mark.parametrize("optimizer", ["adamw", "muon", "dion"])
def test_it_trains_with_every_optimizer(optimizer, tiny_tokenizer, tmp_path):
    from hale_llm.training.lean_trainer import train_lean

    cfg = _train_cfg(tmp_path, steps=30)
    cfg.train.optimizer_type, cfg.train.lean.save_every = optimizer, 0
    losses = train_lean(cfg, tokenizer=tiny_tokenizer)
    assert all(map(torch.isfinite, map(torch.tensor, losses))) and losses[-1] < losses[0]
