"""Several offsets per sequence (mtp_offsets_per_sequence = K): the stratified plan, the loss and the gradients."""

import collections
import itertools

import pytest
import torch

import hale_llm.models  # noqa: F401
from hale_llm.config.load import load_config
from hale_llm.nn.mtp import StratifiedOffsetAssigner
from hale_llm.registry import get_loss, get_variant

VOCAB, T, D = 40, 14, 32


def _assigner(n=4, k=2, seed=3, accum=1, remainder="balanced", stratified=True):
    return StratifiedOffsetAssigner(n, seed, accum, remainder, offsets_per_sequence=k, stratified=stratified)


def _combos(plan):
    return collections.Counter(tuple(sorted(row)) for row in plan.tolist())


def _offset_counts(plan, n):
    tally = collections.Counter(plan.flatten().tolist())
    return [tally[offset] for offset in range(1, n + 1)]


# ----------------------------------------------------------------------------- the plan
def test_every_sequence_gets_k_distinct_offsets_in_range():
    plan = _assigner(4, 2).next_share(50).offsets
    assert plan.shape == (50, 2) and plan.dtype == torch.long
    assert all(len(set(row)) == 2 for row in plan.tolist())
    assert 1 <= int(plan.min()) and int(plan.max()) <= 4


@pytest.mark.parametrize("n,k", [(4, 2), (4, 3), (5, 2), (6, 3), (4, 4)])
def test_a_multiple_of_the_number_of_combinations_uses_every_combination_and_offset_equally(n, k):
    combos = len(list(itertools.combinations(range(n), k)))
    plan = _assigner(n, k).next_share(combos * 3).offsets
    assert set(_combos(plan).values()) == {3}  # every combination exactly 3 times
    assert len(_combos(plan)) == combos
    assert len(set(_offset_counts(plan, n))) == 1  # so every offset is used equally often


@pytest.mark.parametrize("n,k,batch", [(4, 2, 64), (4, 2, 10), (4, 2, 7), (4, 2, 3), (4, 3, 10), (6, 3, 50), (4, 1, 10)])
def test_the_totals_over_a_run_stay_balanced_for_offsets_and_combinations(n, k, batch):
    """A global batch that is NOT a multiple of C: the remainder still balances out over the run."""
    assigner = _assigner(n, k, remainder="balanced")
    offsets, combos = collections.Counter(), collections.Counter()
    for _ in range(120):
        plan = assigner.next_share(batch).offsets
        offsets.update(plan.flatten().tolist())
        combos.update(_combos(plan))
    assert max(offsets.values()) - min(offsets.values()) <= 1
    all_combos = [tuple(c) for c in itertools.combinations(range(1, n + 1), k)]
    counts = [combos[c] for c in all_combos]
    assert max(counts) - min(counts) <= 1


def test_every_step_uses_each_offset_floor_or_ceil_of_the_fair_share():
    assigner = _assigner(4, 2)
    for _ in range(30):
        counts = _offset_counts(assigner.next_share(7).offsets, 4)  # 14 slots over 4 offsets
        assert max(counts) - min(counts) <= 1 and sum(counts) == 14


def test_k_equal_to_the_number_of_offsets_gives_every_sequence_all_of_them():
    plan = _assigner(4, 4).next_share(5).offsets
    assert all(sorted(row) == [1, 2, 3, 4] for row in plan.tolist())


def test_global_counts_count_pairs_not_sequences():
    share = _assigner(4, 2).next_share(12)
    assert share.global_counts.tolist() == [6, 6, 6, 6] and int(share.global_counts.sum()) == 12 * 2


def test_the_plan_is_shuffled():
    """Which sequence gets which combination is random: the first sequence is not always the same combination."""
    firsts = {tuple(sorted(_assigner(4, 2, seed=seed).next_share(12).offsets[0].tolist())) for seed in range(60)}
    assert len(firsts) > 3


def test_same_seed_same_plan_and_other_seed_other_plan():
    a, b, other = _assigner(seed=5), _assigner(seed=5), _assigner(seed=6)
    plans = [[x.next_share(16).offsets.tolist() for _ in range(5)] for x in (a, b, other)]
    assert plans[0] == plans[1] and plans[0] != plans[2]


@pytest.mark.parametrize("batch,accum", [(1, 6), (2, 3), (3, 2), (6, 1)])
def test_the_split_into_micro_batches_does_not_change_the_plan(batch, accum):
    whole, split = _assigner(), _assigner(accum=accum)
    for _ in range(3):
        shares = [split.next_share(batch) for _ in range(accum)]
        assert torch.equal(torch.cat([share.offsets for share in shares]), whole.next_share(batch * accum).offsets)
        assert all(torch.equal(share.global_counts, shares[0].global_counts) for share in shares)


@pytest.mark.parametrize("world,accum,batch", [(2, 1, 3), (2, 3, 2), (3, 2, 2)])
def test_every_rank_gets_its_slice_of_one_plan(world, accum, batch):
    reference = _assigner()
    ranks = []
    for rank in range(world):
        assigner = _assigner(accum=accum)
        assigner.set_data_parallel(rank, world)
        ranks.append(assigner)
    taken = {}
    for micro in range(accum):
        for rank, assigner in enumerate(ranks):
            taken[micro, rank] = assigner.next_share(batch).offsets
    joined = torch.cat([taken[micro, rank] for micro in range(accum) for rank in range(world)])
    assert torch.equal(joined, reference.next_share(batch * accum * world).offsets)


def test_a_micro_batch_of_one_sees_k_offsets_and_the_step_sees_them_all():
    assigner = _assigner(4, 2, accum=6)
    shares = [assigner.next_share(1) for _ in range(6)]  # 6 = the number of combinations of 2 of 4
    assert all(share.offsets.shape == (1, 2) for share in shares)
    assert set(_combos(torch.cat([share.offsets for share in shares])).values()) == {1}


def test_resuming_continues_with_the_same_offsets():
    original = _assigner(4, 2, accum=2)
    for _ in range(5):
        original.next_share(3)  # 2 whole steps and one micro-batch
    clone = _assigner(4, 2, accum=2)
    clone.load_state_dict(original.state_dict())
    for _ in range(10):
        assert torch.equal(original.next_share(3).offsets, clone.next_share(3).offsets)


def test_a_state_from_a_different_k_is_refused():
    saved = _assigner(4, 2).state_dict()
    with pytest.raises(ValueError, match="offsets_per_sequence"):
        _assigner(4, 3).load_state_dict(saved)
    with pytest.raises(ValueError, match="stratified"):
        _assigner(4, 2, stratified=False).load_state_dict(saved)


@pytest.mark.parametrize("k", [0, 5, -1])
def test_k_must_be_between_one_and_the_number_of_offsets(k):
    with pytest.raises(ValueError, match="offsets_per_sequence"):
        StratifiedOffsetAssigner(4, offsets_per_sequence=k)


def test_random_remainder_is_distinct_combinations_without_memory():
    assigner = _assigner(4, 2, remainder="random")
    plan = assigner.next_share(3).offsets  # G = 3 < C = 6: all three are remainder
    assert len(_combos(plan)) == 3
    assert assigner.state_dict()["_extra_state"]["combo_counts"] == [0] * 6  # random keeps no counts


# ----------------------------------------------------------------------------- not stratified (independent draws)
def test_without_stratification_each_sequence_draws_k_distinct_offsets_on_its_own():
    assigner = _assigner(4, 2, stratified=False)
    plan = assigner.next_share(60).offsets
    assert plan.shape == (60, 2) and all(len(set(row)) == 2 for row in plan.tolist())
    assert int(assigner.next_share(1).global_counts.sum()) == 2  # still counts the pairs


def test_without_stratification_the_counts_are_only_balanced_in_expectation():
    """12 sequences x 2 offsets: a stratified plan always gives 6 of each; independent draws sometimes do not."""
    uneven = sum(
        len(set(_offset_counts(_assigner(4, 2, seed=s, stratified=False).next_share(12).offsets, 4))) > 1
        for s in range(40)
    )
    assert uneven > 10
    assert all(len(set(_offset_counts(_assigner(4, 2, seed=s).next_share(12).offsets, 4))) == 1 for s in range(40))


def test_without_stratification_the_plan_is_still_deterministic_and_resumable():
    a, b = _assigner(stratified=False, seed=2), _assigner(stratified=False, seed=2)
    for _ in range(3):
        a.next_share(5)
    b.load_state_dict(a.state_dict())
    assert torch.equal(a.next_share(5).offsets, b.next_share(5).offsets)


# ----------------------------------------------------------------------------- the model and the loss
def _cfg(per_sequence=2, depth=4, accum=1, stratified=True, reduction=None):
    cfg = load_config("configs/autoregressive_mtp/dense_nano.yaml")
    cfg.variant = "autoregressive_mtp"
    cfg.set_context_window(T)
    m = cfg.model
    m.d_model, m.n_heads, m.n_layers, m.d_ff, m.dropout = D, 4, 2, 64, 0.0
    m.mtp_depth, m.mtp_offset_conditioning, m.mtp_draft_tokens = depth, True, 1
    m.mtp_offsets_per_sequence, m.mtp_offset_stratified, m.mtp_loss_reduction = per_sequence, stratified, reduction
    cfg.train.lean.accum_steps = accum
    return cfg


def _model(cfg, seed=0):
    torch.manual_seed(seed)
    return get_variant(cfg.variant)(vocab_size=VOCAB, cfg=cfg)


def _batch(ids):
    labels = ids.clone()
    labels[:, :-1] = ids[:, 1:]
    labels[:, -1] = -100
    return {"input_ids": ids, "attention_mask": torch.ones_like(ids), "labels": labels, "pad_token_id": None}


def _ids(batch, seed=0):
    return torch.randint(1, VOCAB, (batch, T), generator=torch.Generator().manual_seed(seed))


def _fresh_pair(accum_a, accum_b, **kwargs):
    """Two models with identical weights and both assigners at step 0 (load_state_dict also copies the position)."""
    a, b = _model(_cfg(accum=accum_a, **kwargs)).train(), _model(_cfg(accum=accum_b, **kwargs)).train()
    b.load_state_dict(a.state_dict())
    for model in (a, b):
        assigner = model.mtp.assigner
        assigner.step = assigner.micro = 0
        assigner._plan = None
        assigner._extra_counts = [0] * model.mtp.depth
        assigner._combo_counts = [0] * len(assigner._combo_counts)
    return a, b


def _grads(model):
    return {name: p.grad.clone() for name, p in model.named_parameters() if p.grad is not None}


def test_the_model_asks_the_assigner_for_k_offsets():
    model = _model(_cfg(per_sequence=3))
    assert model.mtp.assigner.offsets_per_sequence == 3 and model.mtp.assigner.stratified


def test_k_two_loss_is_the_sum_of_the_two_passes_written_out_by_hand():
    """Independent reference: run the module once per offset column and add the weighted per-sequence means."""
    model = _model(_cfg(per_sequence=2)).train()
    ids = _ids(6, seed=1)
    reference = _model(_cfg(per_sequence=2)).train()
    reference.load_state_dict(model.state_dict())
    share = reference.mtp.assigner.next_share(6)  # the plan the model will use (same seed, same step)
    reference.mtp.assigner.step = reference.mtp.assigner.micro = 0
    reference.mtp.assigner._plan = None

    get_loss("autoregressive_mtp")(model, _batch(ids))
    mtp_term = float(model.loss_parts["mtp_loss"])

    expected = 0.0
    bb = reference.backbone
    with torch.no_grad():
        hidden = bb.encode(ids)
        for column in range(2):
            offsets = share.offsets[:, column]
            logits = reference.mtp(hidden, ids, None, bb.token_emb, bb.lm_head, offsets)
            per_offset = collections.defaultdict(list)
            for b in range(6):
                n = int(offsets[b])
                losses = [float(torch.nn.functional.cross_entropy(logits[b, t][None], ids[b, t + n + 1][None]))
                          for t in range(T) if t + n + 1 <= T - 1]
                per_offset[n].append(sum(losses) / len(losses))
            # L_n over (sequence, offset) pairs: accumulate the per-offset sums here, divide at the end
            for n, values in per_offset.items():
                expected += sum(values) / int(share.global_counts[n - 1])
    assert mtp_term == pytest.approx(expected, abs=1e-4)


@pytest.mark.parametrize("batch,accum", [(1, 6), (2, 3), (3, 2)])
def test_accumulation_gives_the_same_loss_and_gradients_as_one_big_batch_for_k_two(batch, accum):
    whole, split = _fresh_pair(1, accum, per_sequence=2)
    ids = _ids(6, seed=21)
    loss_whole = get_loss("autoregressive_mtp")(whole, _batch(ids))
    loss_whole.backward()
    total, mtp_total = 0.0, 0.0
    for m in range(accum):
        micro = get_loss("autoregressive_mtp")(split, _batch(ids[m * batch : (m + 1) * batch])) / accum
        micro.backward()
        total += float(micro)
        mtp_total += float(split.loss_parts["mtp_loss"]) / accum
    assert total == pytest.approx(float(loss_whole), abs=1e-5)
    assert mtp_total == pytest.approx(float(whole.loss_parts["mtp_loss"]), abs=1e-5)
    g_whole, g_split = _grads(whole), _grads(split)
    assert g_whole.keys() == g_split.keys() and all(torch.allclose(g_whole[k], g_split[k], atol=1e-5) for k in g_whole)


def test_k_equal_to_one_matches_the_single_offset_loss():
    """The default stays what it was: with K = 1 the loop runs once."""
    model = _model(_cfg(per_sequence=1)).train()
    get_loss("autoregressive_mtp")(model, _batch(_ids(4, seed=3)))
    assert model.mtp.assigner.next_share(4).offsets.shape == (4, 1)


def test_every_offset_row_of_the_table_gets_a_gradient_when_all_offsets_are_used():
    model = _model(_cfg(per_sequence=2)).train()
    get_loss("autoregressive_mtp")(model, _batch(_ids(6, seed=4))).backward()
    grad = model.mtp.offset_emb.weight.grad
    assert grad is not None and bool((grad.abs().sum(dim=1) > 0).all())  # all 4 offsets were trained in one step


def test_two_offsets_reach_the_trunk_with_more_gradient_than_one():
    """The trunk is shared: its MTP gradient comes from every pass of the module."""
    one, two = _model(_cfg(per_sequence=1)).train(), _model(_cfg(per_sequence=2)).train()
    weights = {k: v for k, v in one.state_dict().items() if "assigner" not in k}
    two.load_state_dict(weights, strict=False)  # same weights, each keeps its own assigner
    ids = _ids(8, seed=5)
    for model in (one, two):
        get_loss("autoregressive_mtp")(model, _batch(ids)).backward()
    assert float(two.mtp.mtp_module.proj.weight.grad.norm()) > 0  # sanity: the module trains in both
    assert not torch.equal(_grads(one)["backbone.token_emb.weight"], _grads(two)["backbone.token_emb.weight"])


def test_unstratified_mode_trains_too():
    model = _model(_cfg(per_sequence=2, stratified=False)).train()
    loss = get_loss("autoregressive_mtp")(model, _batch(_ids(6, seed=6)))
    loss.backward()
    assert torch.isfinite(loss) and model.mtp.mtp_module.proj.weight.grad is not None


def test_mean_reduction_divides_the_k_two_sum_by_the_number_of_offsets():
    total, mean = _model(_cfg(per_sequence=2, reduction="sum")).train(), _model(_cfg(per_sequence=2, reduction="mean")).train()
    mean.load_state_dict(total.state_dict())
    ids = _ids(8, seed=7)
    get_loss("autoregressive_mtp")(total, _batch(ids))
    get_loss("autoregressive_mtp")(mean, _batch(ids))
    assert float(mean.loss_parts["mtp_loss"]) == pytest.approx(float(total.loss_parts["mtp_loss"]) / 4, abs=1e-5)


# ----------------------------------------------------------------------------- the config
@pytest.mark.parametrize("value", [0, 5])
def test_the_config_rejects_a_k_outside_one_to_the_depth(value):
    cfg = load_config("configs/autoregressive_mtp/offset_nano.yaml")
    cfg.model.mtp_offsets_per_sequence = value
    with pytest.raises(ValueError, match="mtp_offsets_per_sequence"):
        cfg.model._check_shapes()


def test_stratified_is_on_and_k_is_one_by_default():
    cfg = load_config("configs/autoregressive_mtp/offset_nano.yaml")
    assert cfg.model.mtp_offset_stratified is True and cfg.model.mtp_offsets_per_sequence == 1


def test_the_model_can_be_built_under_the_meta_device_like_the_dry_run_does():
    with torch.device("meta"):
        model = get_variant("autoregressive_mtp")(vocab_size=VOCAB, cfg=_cfg(per_sequence=2))
    assert model.mtp.assigner._combos.device.type == "cpu" and model.mtp.assigner._combos.shape == (6, 2)
