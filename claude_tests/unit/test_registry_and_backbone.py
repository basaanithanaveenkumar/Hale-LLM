import torch

import hale_llm.models  # noqa: F401
from hale_llm.config.sections.model import ModelConfig
from hale_llm.nn.backbones import TransformerBackbone
from hale_llm.optim import OPTIMIZERS, build_optimizer
from hale_llm.registry import MODELS, get_variant
from hale_llm.runtime import format_model_summary


def test_registry_has_all_variants():
    assert set(MODELS) == {"autoregressive", "autoregressive_moe", "autoregressive_mtp", "autoregressive_moe_mtp"}


def test_get_variant_unknown_lists_options():
    try:
        get_variant("nope")
        assert False
    except KeyError as error:
        assert "autoregressive" in str(error)


def test_backbone_forward_shape():
    cfg = ModelConfig(d_model=16, n_heads=4, n_layers=1, d_ff=32, max_length=8)
    model = TransformerBackbone(32, cfg)
    assert model(torch.randint(0, 32, (2, 8))).shape == (2, 8, 32)


def test_model_summary_lists_modules():
    class Wrapper(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Linear(4, 2))

    text = format_model_summary(Wrapper(), title="model summary")
    assert "TOTAL" in text and "Trainable" in text and "backbone" in text


def test_optimizer_registry_has_builtins():
    assert {"adam", "adamw", "sgd", "muon", "dion"} <= set(OPTIMIZERS)


def test_build_adamw_has_torch_api():
    model = torch.nn.Linear(4, 2)
    opt = build_optimizer("adamw", model, {"lr": 1e-3, "weight_decay": 0.0})
    loss = model(torch.ones(3, 4)).sum()
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
    assert "lr" in opt.param_groups[0]
    opt.load_state_dict(opt.state_dict())
