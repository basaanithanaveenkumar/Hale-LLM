"""Dion wiring: selection, parameter split, the update itself, learning-rate scaling, checkpoint round trip."""

from __future__ import annotations

import copy
import sys

import pytest
import torch
from torch import nn

from hale_llm.config.sections.train import TrainConfig
from hale_llm.optim import OPTIMIZERS, build_optimizer
from hale_llm.optim.parameter_groups import split_parameters


class TinyLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_emb = nn.Embedding(20, 8)
        self.proj = nn.Linear(8, 8, bias=True)
        self.norm = nn.LayerNorm(8)
        self.lm_head = nn.Linear(8, 20, bias=False)

    def forward(self, ids):
        return self.lm_head(self.norm(self.proj(self.token_emb(ids))))


@pytest.fixture(autouse=True)
def _fresh_compile_cache():
    """dion compiles its update with torch.compile, which allows only 8 recompiles per process. A session that has
    built many Dion optimizers falls back to eager, which rounds differently in the last bits; start each test clean
    so the exact comparisons below do not depend on what ran before."""
    torch._dynamo.reset()


CONFIG = {"lr": 1e-3, "weight_decay": 0.1, "dion_lr": 0.02, "dion_mu": 0.95, "dion_rank": 768,
          "adam_beta1": 0.9, "adam_beta2": 0.95}


def _backward(model, seed=0):
    torch.manual_seed(seed)
    model(torch.randint(0, 20, (4, 6))).pow(2).mean().backward()


# ----- selection -----
def test_dion_is_registered_and_is_not_the_default():
    assert "dion" in OPTIMIZERS
    assert TrainConfig().optimizer_type == "adamw"
    assert (TrainConfig().dion_lr, TrainConfig().dion_mu, TrainConfig().dion_rank) == (0.02, 0.95, 768)


def test_the_config_can_name_it():
    assert TrainConfig(optimizer_type="dion").optimizer_type == "dion"


# ----- the split -----
def test_only_hidden_matrices_go_to_dion_and_the_logged_lr_is_the_adamw_one():
    model = TinyLM()
    opt = build_optimizer("dion", model, CONFIG)
    by_algorithm = {g["algorithm"]: g for g in opt.param_groups}
    assert by_algorithm["dion"]["params"] == [model.proj.weight]
    assert len(by_algorithm["adamw"]["params"]) == 5  # embedding, head, bias, norm gain and bias
    assert opt.param_groups[0]["algorithm"] == "adamw" and opt.param_groups[0]["lr"] == 1e-3
    assert by_algorithm["dion"]["lr"] == 0.02 and by_algorithm["dion"]["mu"] == 0.95
    assert by_algorithm["dion"]["weight_decay"] == 0.1 and by_algorithm["adamw"]["weight_decay"] == 0.0


def test_the_split_agrees_with_the_one_muon_uses():
    model = TinyLM()
    hidden, others = split_parameters(model.named_parameters())
    groups = {g["algorithm"]: g["params"] for g in build_optimizer("dion", model, CONFIG).param_groups}
    assert groups["dion"] == hidden and groups["adamw"] == others


def test_a_model_without_hidden_matrices_is_refused():
    only_embedding = nn.ModuleDict({"token_emb": nn.Embedding(5, 4)})  # a lookup table is not a hidden matrix
    with pytest.raises(ValueError, match="no 2-D hidden weights"):
        build_optimizer("dion", only_embedding, CONFIG)


def test_a_missing_package_says_how_to_install_it(monkeypatch):
    monkeypatch.setitem(sys.modules, "dion", None)  # makes `from dion import ...` raise ImportError
    with pytest.raises(ImportError, match="pip install 'dion @ git\\+https://github.com/microsoft/dion.git'"):
        build_optimizer("dion", TinyLM(), CONFIG)


# ----- the update, against the formula written out from the paper -----
def _reference_dion_step(weight, grad, q, lr, mu, weight_decay):
    """One Dion step from scratch: momentum, power iteration, QR, error feedback, column-normalised Q, scaled update."""
    momentum = grad.clone()  # momentum starts at zero, then the gradient is added
    p = momentum @ q
    p, _ = torch.linalg.qr(p)
    r = momentum.T @ p
    new_q = r / (r.norm(dim=0, keepdim=True) + 1e-8)
    scaled_lr = lr * (weight.size(0) / weight.size(1)) ** 0.5
    return weight * (1 - weight_decay * lr) - scaled_lr * (p @ new_q.T)


@pytest.mark.parametrize("rank", [768, 3])  # 768 is capped at the matrix size; 3 is a real low-rank step
def test_one_step_equals_the_reference_update(rank):
    model = TinyLM()
    opt = build_optimizer("dion", model, {**CONFIG, "dion_rank": rank})
    _backward(model)
    weight, grad = model.proj.weight.detach().clone(), model.proj.weight.grad.clone()
    torch.manual_seed(1234)  # the power-iteration start Q is random; fix it, then rebuild it for the reference
    opt.step()
    torch.manual_seed(1234)
    q = torch.randn(8, min(rank, 8))
    expected = _reference_dion_step(weight, grad, q, lr=0.02, mu=0.95, weight_decay=0.1)
    assert torch.allclose(model.proj.weight, expected, atol=1e-5)


@pytest.mark.parametrize("rank", [3, 8])
def test_the_update_of_a_matrix_has_rank_at_most_dion_rank(rank):
    """The step is P Q^T with P and Q of ``rank`` columns, so its rank is the point of the method."""
    model = TinyLM()
    opt = build_optimizer("dion", model, {**CONFIG, "weight_decay": 0.0, "dion_rank": rank})
    _backward(model)
    before = model.proj.weight.detach().clone()
    opt.step()
    assert torch.linalg.matrix_rank(before - model.proj.weight.detach(), atol=1e-7).item() == rank


# ----- the other parameters and the schedule -----
def test_step_updates_every_parameter_including_adamw_ones():
    model = TinyLM()
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    opt = build_optimizer("dion", model, CONFIG)
    _backward(model)
    opt.step()
    assert all(not torch.equal(before[n], p) for n, p in model.named_parameters())


def test_embeddings_norms_and_biases_get_no_weight_decay():
    model = TinyLM()
    opt = build_optimizer("dion", model, {**CONFIG, "lr": 0.0})  # AdamW lr 0: only decay could move them
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.grad = torch.zeros_like(parameter)
    before = {n: p.detach().clone() for n, p in model.named_parameters() if n != "proj.weight"}
    opt.step()
    assert all(torch.equal(before[n], dict(model.named_parameters())[n]) for n in before)


def test_the_trainer_schedule_scales_both_groups():
    """The trainer sets group['lr'] = base_lr * scale each step; a scale of 0 must freeze every parameter."""
    model = TinyLM()
    opt = build_optimizer("dion", model, CONFIG)
    for group in opt.param_groups:
        group["base_lr"] = group["lr"]
        group["lr"] = group["base_lr"] * 0.0
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    _backward(model)
    opt.step()
    assert all(torch.equal(before[n], p) for n, p in model.named_parameters())
    for group in opt.param_groups:
        group["lr"] = group["base_lr"] * 0.5
    opt.step()
    assert any(not torch.equal(before[n], p) for n, p in model.named_parameters())


# ----- learning and resuming -----
def test_dion_trains_a_tiny_model():
    torch.manual_seed(0)
    model = TinyLM()
    opt = build_optimizer("dion", model, {**CONFIG, "lr": 1e-2, "weight_decay": 0.0})
    ids = torch.randint(0, 20, (16, 6))
    target = torch.randint(0, 20, (16, 6))
    losses = []
    for _ in range(150):
        opt.zero_grad()
        loss = nn.functional.cross_entropy(model(ids).reshape(-1, 20), target.reshape(-1))
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < 0.6 * losses[0], (losses[0], losses[-1])  # AdamW reaches ~0.45 on this task


def test_state_dict_round_trip_continues_exactly():
    torch.manual_seed(0)
    model = TinyLM()
    opt = build_optimizer("dion", model, CONFIG)
    _backward(model, seed=1)
    opt.step()
    clone = TinyLM()
    clone.load_state_dict(model.state_dict())
    restored = build_optimizer("dion", clone, CONFIG)
    # A real resume loads the state from disk; deepcopy gives the restored optimizer its own tensors (load_state_dict
    # alone would share the live ones, and the first optimizer's next step would change the second's state).
    restored.load_state_dict(copy.deepcopy(opt.state_dict()))
    for target_model, optimizer in ((model, opt), (clone, restored)):
        optimizer.zero_grad()
        _backward(target_model, seed=2)
        optimizer.step()
    for (name, a), (_, b) in zip(model.named_parameters(), clone.named_parameters()):
        assert torch.equal(a, b), name


# ----- from the YAML config to the optimizer -----
def test_the_trainer_hands_the_dion_settings_from_the_config_to_the_optimizer(tiny_tokenizer, tmp_path, monkeypatch):
    from hale_llm.config.schema import load_config
    from hale_llm.training import lean_trainer

    cfg = load_config("configs/autoregressive/nano.yaml")
    cfg.train.steps, cfg.train.epochs, cfg.train.batch_size, cfg.train.resume = 2, None, 2, False
    cfg.train.checkpoint_path = str(tmp_path / "run" / "model.pt")
    cfg.train.lean.save_every, cfg.train.lean.eval_every, cfg.train.lean.keep_best, cfg.train.lean.compile = 0, 0, 0, False
    cfg.model.d_model, cfg.model.n_heads, cfg.model.n_layers, cfg.model.d_ff = 32, 4, 1, 64
    cfg.data.source, cfg.data.overfit_text, cfg.data.n_overfit_copies = "overfit", "the cat sat on the mat", 16
    cfg.set_context_window(16)
    cfg.train.optimizer_type, cfg.train.dion_lr, cfg.train.dion_mu, cfg.train.dion_rank = "dion", 0.123, 0.77, 5
    built = []
    real_build = lean_trainer.build_optimizer

    def spy(name, model, config):
        optimizer = real_build(name, model, config)
        built.append((name, optimizer))
        return optimizer

    monkeypatch.setattr(lean_trainer, "build_optimizer", spy)
    lean_trainer.train_lean(cfg, tokenizer=tiny_tokenizer)
    name, optimizer = built[0]
    dion_group = next(g for g in optimizer.param_groups if g["algorithm"] == "dion")
    assert name == "dion" and dion_group["mu"] == 0.77 and dion_group["base_lr"] == 0.123
    assert optimizer.optimizer.rank == 5


# ----- many different parameter shapes in one run -----
def test_a_model_with_many_distinct_parameter_shapes_does_not_hit_the_compile_limit(monkeypatch):
    """dion's AdamW update is compiled with fullgraph=True, and PyTorch's default of 8 recompiles turned the 9th distinct
    parameter shape into a crash in the middle of training. Norm gains, biases, embeddings and routers easily give 8+."""
    sizes = [8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52]  # 12 distinct 1-D and 2-D shapes on the AdamW side

    class WideZoo(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(8, 8)  # the Dion matrix
            self.biases = nn.ParameterList([nn.Parameter(torch.randn(n)) for n in sizes])
            self.token_emb = nn.ParameterList([nn.Parameter(torch.randn(n, 8)) for n in sizes[:6]])

    torch._dynamo.reset()
    monkeypatch.setattr(torch._dynamo.config, "recompile_limit", 8)  # PyTorch's default, as in a fresh process
    model = WideZoo()
    opt = build_optimizer("dion", model, CONFIG)
    for parameter in model.parameters():
        parameter.grad = torch.randn_like(parameter)
    opt.step()  # raised FailOnRecompileLimitHit before the limit was raised
    assert torch._dynamo.config.recompile_limit >= 64


def test_building_the_optimizer_never_lowers_a_higher_limit(monkeypatch):
    monkeypatch.setattr(torch._dynamo.config, "recompile_limit", 500)
    build_optimizer("dion", TinyLM(), CONFIG)
    assert torch._dynamo.config.recompile_limit == 500
