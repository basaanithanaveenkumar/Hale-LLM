"""Muon wiring: parameter split, stepping, checkpoint round-trip and config selection."""

import pytest
import torch
from torch import nn

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


CONFIG = {"lr": 1e-3, "weight_decay": 0.0, "muon_lr": 0.02, "muon_momentum": 0.95}


def test_muon_is_registered():
    assert "muon" in OPTIMIZERS


def test_only_hidden_matrices_go_to_muon():
    model = TinyLM()
    muon_params, adamw_params = split_parameters(model.named_parameters())
    assert muon_params == [model.proj.weight]
    # embedding, head, bias and norm gain/bias all stay on AdamW
    assert len(adamw_params) == 5


def test_step_updates_every_parameter():
    model = TinyLM()
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    opt = build_optimizer("muon", model, CONFIG)
    model(torch.randint(0, 20, (4, 6))).sum().backward()
    opt.step()
    assert all(not torch.equal(before[n], p) for n, p in model.named_parameters())


def test_logged_learning_rate_is_the_adamw_one():
    opt = build_optimizer("muon", TinyLM(), CONFIG)
    assert opt.param_groups[0]["lr"] == 1e-3


def test_state_dict_round_trip():
    model = TinyLM()
    opt = build_optimizer("muon", model, CONFIG)
    model(torch.randint(0, 20, (2, 4))).sum().backward()
    opt.step()
    restored = build_optimizer("muon", model, CONFIG)
    restored.load_state_dict(opt.state_dict())  # must not raise


def test_model_without_matrices_is_rejected():
    with pytest.raises(ValueError, match="2-D"):
        build_optimizer("muon", nn.Sequential(nn.LayerNorm(4)), CONFIG)


def test_adamw_is_the_llm_recipe():
    model = TinyLM()
    opt = build_optimizer("adamw", model, {**CONFIG, "weight_decay": 0.1})
    assert isinstance(opt.optimizer, torch.optim.AdamW)
    assert opt.optimizer.defaults["betas"] == (0.9, 0.95)
    decayed, undecayed = opt.param_groups
    assert decayed["weight_decay"] == 0.1 and decayed["params"] == [model.proj.weight]
    assert undecayed["weight_decay"] == 0.0  # embedding, head, bias, norm


def test_sgd_and_adam_are_registered():
    opt = build_optimizer("sgd", TinyLM(), {"lr": 0.1})
    assert isinstance(opt.optimizer, torch.optim.SGD)
