"""The default optimizer settings are the vanilla LLM recipe: AdamW, betas 0.9 / 0.95, eps 1e-8, decay 0.1, clip 1."""

import torch

from hale_llm.config.load import load_config
from hale_llm.config.sections.train import TrainConfig
from hale_llm.optim import build_optimizer


def test_the_schema_defaults_are_the_vanilla_recipe():
    train = TrainConfig()
    assert train.optimizer_type == "adamw"
    assert (train.adam_beta1, train.adam_beta2, train.adam_eps) == (0.9, 0.95, 1e-8)
    assert train.weight_decay == 0.1 and train.grad_clip == 1.0


def test_the_base_config_keeps_them():
    cfg = load_config("configs/autoregressive/nano.yaml")
    assert cfg.train.weight_decay == 0.1 and cfg.train.grad_clip == 1.0 and cfg.train.adam_eps == 1e-8


def test_the_optimizer_uses_them_when_nothing_is_set():
    model = torch.nn.Linear(4, 2)
    optimizer = build_optimizer("adamw", model, {"lr": 1e-3})
    group = optimizer.param_groups[0]
    assert group["betas"] == (0.9, 0.95) and group["eps"] == 1e-8 and group["weight_decay"] == 0.1


def test_the_epsilon_is_configurable():
    optimizer = build_optimizer("adamw", torch.nn.Linear(4, 2), {"lr": 1e-3, "adam_eps": 1e-6})
    assert optimizer.param_groups[0]["eps"] == 1e-6
