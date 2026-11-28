"""The looped (Parcae) variant: stable injection, per-sequence depths, truncated backprop, causality, training."""

import math

import pytest
import torch

import hale_llm.models  # noqa: F401
from hale_llm.config.load import load_config
from hale_llm.nn.looped import LoopedBackbone, RecurrenceSampler, StableInjection
from hale_llm.optim.parameter_groups import is_hidden_matrix
from hale_llm.registry import get_loss, get_variant

VOCAB, S, D = 40, 12, 32


def _cfg(prelude=1, recurrent=2, coda=1, mean=4, backprop=None, sampling="fixed", ffn="mlp", decay=0.5):
    cfg = load_config("configs/autoregressive_moe_looped/nano.yaml")
    cfg.set_context_window(S)
    m = cfg.model
    m.d_model, m.n_heads, m.d_ff, m.dropout, m.ffn_type = D, 4, 64, 0.0, ffn
    m.moe_hidden_dim, m.moe_router = 48, "topk"
    m.n_layers = prelude + recurrent + coda
    m.loop_prelude_layers, m.loop_recurrent_layers, m.loop_coda_layers = prelude, recurrent, coda
    m.loop_mean_recurrence, m.loop_mean_backprop, m.loop_sampling, m.loop_decay_init = mean, backprop, sampling, decay
    return cfg


def _model(seed=0, **kwargs):
    cfg = _cfg(**kwargs)
    torch.manual_seed(seed)
    return get_variant(cfg.variant)(vocab_size=VOCAB, cfg=cfg)


def _ids(batch=3, seed=0):
    return torch.randint(0, VOCAB, (batch, S), generator=torch.Generator().manual_seed(seed))


# ----------------------------------------------------------------------------- the stable injection
def test_the_state_decay_is_inside_zero_one_at_the_start_and_equals_the_requested_value():
    injection = StableInjection(8, decay_init=0.3)
    assert torch.allclose(injection.state_decay(), torch.full((8,), 0.3), atol=1e-6)
    assert float(injection.spectral_radius()) == pytest.approx(0.3, abs=1e-6)


def test_the_injection_matches_the_formula_written_out_by_hand():
    """A = -exp(log_A), A_bar = exp(Delta * A), B_bar = Delta * B (row i scaled by Delta_i)."""
    torch.manual_seed(0)
    injection = StableInjection(6, decay_init=0.6)
    with torch.no_grad():
        injection.log_a.copy_(torch.randn(6))
        injection.delta_raw.copy_(torch.randn(6))
        injection.input_matrix.copy_(torch.randn(6, 6))
    h, e = torch.randn(2, 5, 6), torch.randn(2, 5, 6)
    delta = torch.nn.functional.softplus(injection.delta_raw)
    a_bar = torch.exp(delta * (-torch.exp(injection.log_a)))
    b_bar = delta[:, None] * injection.input_matrix
    expected = torch.stack([
        torch.stack([a_bar * h[b, s] + b_bar @ e[b, s] for s in range(5)]) for b in range(2)
    ])
    assert torch.allclose(injection(h, e), expected, atol=1e-6)


@pytest.mark.parametrize("seed", range(5))
def test_the_spectral_radius_stays_below_one_for_ANY_parameter_values(seed):
    """The whole point of Parcae: rho(A_bar) < 1 no matter what the optimizer does to log_A, Delta and B."""
    torch.manual_seed(seed)
    injection = StableInjection(64)
    with torch.no_grad():
        injection.log_a.copy_(torch.randn(64))
        injection.delta_raw.copy_(torch.randn(64))
        injection.input_matrix.copy_(torch.randn(64, 64) * 100)  # B is unconstrained and does not matter for stability
    decay = injection.state_decay()
    assert bool((decay > 0).all()) and float(injection.spectral_radius()) < 1.0  # strictly inside (0, 1)
    with torch.no_grad():  # absurd values: float rounding may give exactly 0 or 1, but never outside [0, 1]
        injection.log_a.copy_(torch.randn(64) * 10)
        injection.delta_raw.copy_(torch.randn(64) * 10)
    assert bool((injection.state_decay() >= 0).all()) and float(injection.spectral_radius()) <= 1.0
    # The linear system h' = A_bar h + B_bar e (R_bar dropped) converges to a finite fixed point.
    e = torch.randn(1, 1, 64)
    h = torch.zeros(1, 1, 64)
    for _ in range(2000):
        h = injection(h, e)
    assert bool(torch.isfinite(h).all())


def test_the_spectral_radius_stays_at_most_one_after_wild_updates_of_every_parameter():
    """Whatever steps the optimizer takes (here: big random kicks, 50 times), A_bar remains in [0, 1]."""
    torch.manual_seed(0)
    injection = StableInjection(16)
    for _ in range(50):
        with torch.no_grad():
            for parameter in injection.parameters():
                parameter.add_(torch.randn_like(parameter) * 3)
        assert bool((injection.state_decay() >= 0).all()) and float(injection.spectral_radius()) <= 1.0


@pytest.mark.parametrize("bad", [0.0, 1.0, -0.1, 1.5])
def test_decay_init_must_be_strictly_between_zero_and_one(bad):
    with pytest.raises(ValueError, match="decay_init"):
        StableInjection(4, decay_init=bad)


def test_every_injection_parameter_gets_a_gradient():
    injection = StableInjection(8)
    injection(torch.randn(2, 3, 8, requires_grad=True), torch.randn(2, 3, 8)).sum().backward()
    assert all(p.grad is not None and bool((p.grad != 0).any()) for p in injection.parameters())


# ----------------------------------------------------------------------------- the depth sampler
def test_fixed_sampling_gives_every_sequence_mu_rec():
    assert RecurrenceSampler(5, mode="fixed").sample(7).tolist() == [5] * 7


def test_poisson_sampling_has_the_right_mean_and_never_goes_below_one():
    torch.manual_seed(0)
    depths = RecurrenceSampler(8).sample(20000)
    assert int(depths.min()) >= 1 and float(depths.float().mean()) == pytest.approx(8.0, abs=0.1)
    assert len(set(depths.tolist())) > 5  # really random per sequence


def test_a_small_mean_still_never_samples_zero_loops():
    torch.manual_seed(0)
    assert int(RecurrenceSampler(1).sample(5000).min()) == 1


def test_mu_bwd_defaults_to_the_ceiling_of_half_of_mu_rec():
    assert [RecurrenceSampler(m).mean_backprop for m in (1, 2, 3, 8, 9)] == [1, 1, 2, 4, 5]
    assert RecurrenceSampler(8, mean_backprop=3).mean_backprop == 3


@pytest.mark.parametrize("kwargs", [{"mean_recurrence": 0}, {"mean_recurrence": 4, "mode": "x"}, {"mean_recurrence": 4, "mean_backprop": 0}])
def test_the_sampler_rejects_bad_settings(kwargs):
    with pytest.raises(ValueError):
        RecurrenceSampler(**kwargs)


# ----------------------------------------------------------------------------- the model against an independent reference
def _reference_logits(model, ids, loops):
    """Plain loops, written from the paper's equations with no masks, no where, no no_grad."""
    bb = model.backbone
    with torch.no_grad():
        x = bb.positional.encode_embeddings(bb.token_emb(ids))
        positions = torch.arange(ids.size(1))
        for block in bb.prelude:
            x = block(x, positions, None, None)
        e = bb.prelude_norm(x)
        h = (bb.eval_state[: ids.size(1)][None].expand(ids.size(0), -1, -1) * bb.state_std)
        a_bar = torch.exp(torch.nn.functional.softplus(bb.injection.delta_raw) * -torch.exp(bb.injection.log_a))
        b_bar = torch.nn.functional.softplus(bb.injection.delta_raw)[:, None] * bb.injection.input_matrix
        for _ in range(loops):
            u = a_bar * h + e @ b_bar.t()  # A_bar h + B_bar e
            for block in bb.recurrent:
                u = block(u, positions, None, None)  # the blocks add R_bar to their input
            h = u
        x = h @ bb.readout.t()
        for block in bb.coda:
            x = block(x, positions, None, None)
        return bb.lm_head(bb.norm(x))


@pytest.mark.parametrize("loops", [1, 2, 5])
def test_eval_logits_equal_the_hand_written_reference_loop(loops):
    model = _model(mean=loops).eval()
    ids = _ids()
    torch.manual_seed(1)
    with torch.no_grad():  # make the injection non-trivial
        model.backbone.injection.log_a.copy_(torch.randn(D) * 0.3)
        model.backbone.injection.delta_raw.copy_(torch.randn(D) * 0.3)
        model.backbone.injection.input_matrix.add_(torch.randn(D, D) * 0.05)
        model.backbone.readout.add_(torch.randn(D, D) * 0.05)
    with torch.no_grad():
        got = model({"input_ids": ids})
    assert torch.allclose(got, _reference_logits(model, ids, loops), atol=1e-5)


def test_num_loops_overrides_the_configured_depth():
    model = _model(mean=3).eval()
    ids = _ids()
    with torch.no_grad():
        default, forced = model({"input_ids": ids}), model({"input_ids": ids, "num_loops": 6})
    assert not torch.allclose(default, forced) and torch.allclose(forced, _reference_logits(model, ids, 6), atol=1e-5)


def test_the_blocks_are_really_reused_the_parameter_count_does_not_grow_with_the_loops():
    assert sum(p.numel() for p in _model(mean=2).parameters()) == sum(p.numel() for p in _model(mean=20).parameters())


def test_the_variant_has_the_same_number_of_distinct_blocks_as_n_layers():
    model = _model(prelude=2, recurrent=3, coda=1)
    assert len(model.backbone.layers) == 6 == model.cfg.model.n_layers


# ----------------------------------------------------------------------------- causality and masks
def test_a_position_never_sees_later_tokens_whatever_the_number_of_loops():
    """The random state and the injection are per position, so the logits at t depend only on tokens 0..t."""
    model = _model(mean=4).eval()
    a = _ids(2, seed=3)
    b = a.clone()
    b[:, 7:] = (b[:, 7:] + 1) % VOCAB  # change the future of position 6
    with torch.no_grad():
        la, lb = model({"input_ids": a}), model({"input_ids": b})
    assert torch.allclose(la[:, :7], lb[:, :7], atol=1e-5) and not torch.allclose(la[:, 8:], lb[:, 8:])


def test_the_logits_of_a_prefix_do_not_change_when_tokens_are_appended():
    """Needed for generation: the fixed evaluation state must not depend on the length of the text."""
    model = _model(mean=3).eval()
    ids = _ids(1, seed=4)
    with torch.no_grad():
        short, long = model({"input_ids": ids[:, :6]}), model({"input_ids": ids})
    assert torch.allclose(short, long[:, :6], atol=1e-5)


def test_evaluation_is_deterministic_and_training_draws_fresh_noise():
    model = _model(mean=3).eval()
    ids = _ids()
    with torch.no_grad():
        assert torch.equal(model({"input_ids": ids}), model({"input_ids": ids}))
    model.train()
    assert not torch.equal(model({"input_ids": ids}), model({"input_ids": ids}))


def test_padding_and_document_masks_reach_the_looped_blocks():
    model = _model(mean=3).eval()
    ids = _ids(1, seed=5)
    docs = torch.tensor([[0] * 6 + [1] * 6])
    other = ids.clone()
    other[:, :6] = (other[:, :6] + 1) % VOCAB  # change document 0 only
    with torch.no_grad():
        a = model({"input_ids": ids, "document_ids": docs})
        b = model({"input_ids": other, "document_ids": docs})
        plain = model({"input_ids": ids})
    assert torch.allclose(a[:, 6:], b[:, 6:], atol=1e-5)  # document 1 does not see document 0
    assert not torch.allclose(a[:, 6:], plain[:, 6:])  # and the mask really changed something


# ----------------------------------------------------------------------------- per-sequence depths and truncated backprop
def _spy_steps(model):
    """Record, for every call of the recurrent step, whether autograd was recording, and which rows moved."""
    calls = []
    original = model.backbone._recurrent_step

    def spy(state, prelude_output, mask_args):
        out = original(state, prelude_output, mask_args)
        calls.append(out.requires_grad)
        return out

    model.backbone._recurrent_step = spy
    return calls


@pytest.mark.parametrize("mean,backprop", [(8, 4), (6, 2), (3, 5), (1, 1)])
def test_only_the_last_mu_bwd_loops_record_a_gradient(mean, backprop):
    model = _model(mean=mean, backprop=backprop).train()
    calls = _spy_steps(model)
    model({"input_ids": _ids()})
    assert len(calls) == mean
    grad_loops = min(mean, backprop)
    assert calls == [False] * (mean - grad_loops) + [True] * grad_loops


def test_early_loops_get_no_gradient_so_the_gradient_equals_a_run_that_starts_from_a_detached_state():
    """Gradient check for the truncation: loops before the last mu_bwd contribute no gradient to anything."""
    mean, backprop = 5, 2
    model = _model(mean=mean, backprop=backprop).train()
    ids = _ids()
    torch.manual_seed(7)
    out = model({"input_ids": ids})
    out.square().mean().backward()
    got = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}

    # Reference: the same computation written with an explicit detach() after the early loops (same noise stream).
    reference = _model(mean=mean, backprop=backprop).train()
    reference.load_state_dict(model.state_dict())
    bb = reference.backbone
    torch.manual_seed(7)
    x = bb.positional.encode_embeddings(bb.token_emb(ids))
    positions = torch.arange(S)
    for block in bb.prelude:
        x = block(x, positions, None, None)
    e = bb.prelude_norm(x)
    bb.sampler.sample(ids.size(0))  # the model drew its depths first, keep the noise stream aligned
    h = torch.randn(ids.size(0), S, D) * bb.state_std
    for t in range(mean):
        u = bb.injection(h, e)
        for block in bb.recurrent:
            u = block(u, positions, None, None)
        h = u.detach() if t < mean - backprop else u
    x = h @ bb.readout.t()
    for block in bb.coda:
        x = block(x, positions, None, None)
    bb.lm_head(bb.norm(x)).square().mean().backward()
    want = {n: p.grad.clone() for n, p in reference.named_parameters() if p.grad is not None}
    assert got.keys() == want.keys()
    assert all(torch.allclose(got[k], want[k], atol=1e-5) for k in got)


def test_per_sequence_depths_in_one_batch_equal_running_each_sequence_alone():
    """Algorithm 2: sequences with fewer loops sit out the first rounds; the result per row must equal its own depth."""
    model = _model(mean=6).eval()
    bb = model.backbone
    ids = _ids(4, seed=6)
    depths = torch.tensor([2, 6, 3, 1])
    with torch.no_grad():
        x = bb.positional.encode_embeddings(bb.token_emb(ids))
        positions = torch.arange(S)
        for block in bb.prelude:
            x = block(x, positions, None, None)
        e = bb.prelude_norm(x)
        h0 = bb._initial_state(4, S, e)
        batched = bb._run_loops(h0, e, depths, (positions, None, None))
        for row in range(4):
            alone = bb._run_loops(h0[row : row + 1], e[row : row + 1], depths[row : row + 1], (positions, None, None))
            assert torch.allclose(batched[row : row + 1], alone, atol=1e-5), row


def test_training_with_random_depths_runs_and_every_parameter_family_gets_a_gradient():
    model = _model(mean=4, sampling="poisson").train()
    loss = get_loss("autoregressive_moe_looped")(model, _batch(_ids(6, seed=8)))
    loss.backward()
    names = {n.split(".")[1] for n, p in model.named_parameters() if p.grad is not None}
    assert {"token_emb", "prelude", "prelude_norm", "injection", "recurrent", "readout", "coda", "norm"} <= names
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def _batch(ids):
    labels = ids.clone()
    labels[:, :-1] = ids[:, 1:]
    labels[:, -1] = -100
    return {"input_ids": ids, "attention_mask": torch.ones_like(ids), "labels": labels, "pad_token_id": None}


# ----------------------------------------------------------------------------- stability in practice
def test_the_hidden_state_stays_bounded_when_looping_far_beyond_the_training_depth():
    """The point of the constraint: many more loops than trained do not blow the state up."""
    model = _model(mean=4).eval()
    ids = _ids()
    norms = []
    bb = model.backbone
    with torch.no_grad():
        x = bb.positional.encode_embeddings(bb.token_emb(ids))
        positions = torch.arange(S)
        for block in bb.prelude:
            x = block(x, positions, None, None)
        e = bb.prelude_norm(x)
        h = bb._initial_state(3, S, e)
        for _ in range(64):
            h = bb._recurrent_step(h, e, (positions, None, None))
            norms.append(float(h.norm(dim=-1).mean()))
    assert all(math.isfinite(n) for n in norms) and max(norms) < 50 * (1 + norms[0])


def test_more_loops_at_test_time_change_the_loss_smoothly_not_explosively():
    model = _model(mean=4).eval()
    batch = _batch(_ids(4, seed=9))
    losses = []
    with torch.no_grad():
        for loops in (1, 2, 4, 8, 16, 32):
            batch["num_loops"] = loops
            losses.append(float(get_loss("autoregressive_moe_looped")(model, batch)))
    assert all(math.isfinite(x) for x in losses) and max(losses) < 3 * math.log(VOCAB)


# ----------------------------------------------------------------------------- configuration and optimizer groups
@pytest.mark.parametrize("n_layers,expected", [(6, (2, 2, 2)), (8, (2, 4, 2)), (12, (4, 4, 4)), (24, (8, 8, 8)), (3, (1, 1, 1))])
def test_the_default_split_loops_the_middle_third_like_the_paper(n_layers, expected):
    """Parcae Table 18: 6 layers -> 2 / 2 / 2, 12 -> 4 / 4 / 4, 24 -> 8 / 8 / 8."""
    cfg = _cfg()
    cfg.model.n_layers = n_layers
    cfg.model.loop_prelude_layers = cfg.model.loop_recurrent_layers = cfg.model.loop_coda_layers = None
    model = get_variant(cfg.variant)(vocab_size=VOCAB, cfg=cfg)
    assert (model.backbone.n_prelude, model.backbone.n_recurrent, model.backbone.n_coda) == expected


@pytest.mark.parametrize("layers", [(1, 1, 1, 5), (0, 2, 1, 3), (1, 2, None, 4)])
def test_a_wrong_layer_split_is_refused(layers):
    prelude, recurrent, coda, n_layers = layers
    cfg = _cfg()
    cfg.model.n_layers = n_layers
    cfg.model.loop_prelude_layers, cfg.model.loop_recurrent_layers, cfg.model.loop_coda_layers = prelude, recurrent, coda
    with pytest.raises(ValueError):
        get_variant(cfg.variant)(vocab_size=VOCAB, cfg=cfg)


def test_the_injection_and_readout_matrices_are_not_treated_as_hidden_matrices_for_muon_or_decay():
    model = _model()
    flags = {n: is_hidden_matrix(n, p) for n, p in model.named_parameters()}
    assert not flags["backbone.injection.input_matrix"] and not flags["backbone.readout"]
    assert flags["backbone.recurrent.0.attn.proj.weight"] and flags["backbone.prelude.0.attn.proj.weight"]


def test_the_model_builds_under_the_meta_device_like_the_dry_run_does():
    cfg = _cfg()
    with torch.device("meta"):
        model = get_variant(cfg.variant)(vocab_size=VOCAB, cfg=cfg)
    assert sum(p.numel() for p in model.parameters()) > 0


def test_a_mixture_of_experts_looped_model_trains():
    model = _model(ffn="moe", mean=3, sampling="poisson").train()
    loss = get_loss("autoregressive_moe_looped")(model, _batch(_ids(4, seed=10)))
    loss.backward()
    assert torch.isfinite(loss) and model.uses_moe


def test_the_state_dict_round_trips_and_gives_the_same_eval_logits():
    a, b = _model(seed=1).eval(), _model(seed=2).eval()
    b.load_state_dict(a.state_dict())
    ids = _ids()
    with torch.no_grad():
        assert torch.allclose(a({"input_ids": ids}), b({"input_ids": ids}))


@pytest.mark.parametrize("field,value", [("loop_decay_init", 1.0), ("loop_mean_recurrence", 0), ("loop_mean_backprop", 0)])
def test_the_config_rejects_bad_loop_settings(field, value):
    cfg = load_config("configs/autoregressive_moe_looped/nano.yaml")
    setattr(cfg.model, field, value)
    with pytest.raises(ValueError):
        cfg.model._check_shapes()


# ----------------------------------------------------------------------------- the trainer end to end
def _train_cfg(tmp_path, steps=60):
    cfg = _cfg(prelude=1, recurrent=1, coda=1, mean=3, sampling="poisson")
    cfg.train.steps, cfg.train.epochs, cfg.train.batch_size, cfg.train.lr = steps, None, 4, 3e-3
    cfg.train.resume, cfg.train.log_every = False, 100
    cfg.train.checkpoint_path = str(tmp_path / "run" / "model.pt")
    cfg.train.lean.save_every, cfg.train.lean.eval_every, cfg.train.lean.keep_best = 0, 0, 0
    cfg.train.lean.warmup_steps, cfg.train.lean.compile, cfg.train.lean.precision = 2, False, "fp32"
    cfg.data.source, cfg.data.n_overfit_copies = "overfit", 12
    cfg.data.overfit_text = "the cat sat on the mat and the dog sat on the log"
    cfg.set_context_window(16)
    return cfg


@pytest.mark.parametrize("optimizer", ["adamw", "muon"])
def test_the_real_trainer_trains_the_looped_model_and_the_loss_falls(optimizer, tiny_tokenizer, tmp_path):
    """Integration: lean trainer, random depths and truncated backprop, AdamW or Muon (injection stays on AdamW)."""
    from hale_llm.training.lean_trainer import train_lean

    cfg = _train_cfg(tmp_path)
    cfg.train.optimizer_type = optimizer
    losses = train_lean(cfg, tokenizer=tiny_tokenizer)
    assert all(math.isfinite(x) for x in losses) and losses[-1] < 0.7 * losses[0]
    state = torch.load(cfg.train.checkpoint_path, weights_only=False)["model_state_dict"]
    assert "backbone.injection.log_a" in state and "backbone.eval_state" not in state  # the fixed state is not saved
    fresh = get_variant(cfg.variant)(vocab_size=state["backbone.token_emb.weight"].size(0), cfg=cfg)
    fresh.load_state_dict(state)  # strict: a checkpoint loads back into a new model


def test_the_spectral_radius_stays_below_one_during_real_training(tiny_tokenizer, tmp_path):
    from hale_llm.training.lean_trainer import train_lean

    cfg = _train_cfg(tmp_path, steps=40)
    cfg.train.lr = 3e-2  # large on purpose
    train_lean(cfg, tokenizer=tiny_tokenizer)
    state = torch.load(cfg.train.checkpoint_path, weights_only=False)["model_state_dict"]
    delta = torch.nn.functional.softplus(state["backbone.injection.delta_raw"])
    decay = torch.exp(delta * -torch.exp(state["backbone.injection.log_a"]))
    assert float(decay.max()) <= 1.0 and bool((decay > 0).all())
