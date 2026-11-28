"""The merged backbone: old checkpoints, one auxiliary-loss function, one block builder, a stable parameter order."""

from __future__ import annotations

import pytest
import torch

import hale_llm.models  # noqa: F401
from hale_llm.config.sections.model import ModelConfig
from hale_llm.nn.backbones import TransformerBackbone
from hale_llm.nn.blocks import TransformerBlock, build_blocks
from hale_llm.nn.moe import DeepseekMoE, auxiliary_loss
from hale_llm.nn.positional import build_positional_encoding


def _cfg(**kw):
    base = dict(d_model=32, n_heads=4, n_layers=3, d_ff=64, max_length=16, norm_type="rms", moe_router="topk")
    return ModelConfig(**{**base, **kw})


def _backbone(seed=0, **kw):
    torch.manual_seed(seed)
    return TransformerBackbone(50, _cfg(**kw))


# ----- checkpoints written before the stack was merged -----
@pytest.mark.parametrize("ffn_type", ["mlp", "moe"])
def test_a_checkpoint_with_the_old_stack_key_names_still_loads(ffn_type):
    new = _backbone(seed=0, ffn_type=ffn_type)
    old_keys = {}
    for key, value in new.state_dict().items():
        old_name = key.replace("layers.", "stack.layers.", 1) if key.startswith("layers.") else (
            "stack." + key if key.startswith("norm.") else key)
        old_keys[old_name] = value.clone()
    assert any(k.startswith("stack.layers.0.") for k in old_keys) and any(k.startswith("stack.norm.") for k in old_keys)

    other = _backbone(seed=99, ffn_type=ffn_type)  # different weights
    ids = torch.randint(0, 50, (2, 16))
    assert not torch.allclose(other.eval()(ids), new.eval()(ids))
    other.load_state_dict(old_keys)  # strict: every old key must find its place
    assert torch.equal(other.eval()(ids), new.eval()(ids))


def test_the_old_names_also_load_when_the_backbone_sits_inside_a_model():
    """The model prefixes every key with ``backbone.``; the load hook must handle the prefix."""
    from hale_llm.config.schema import load_config
    from hale_llm.registry import get_variant

    cfg = load_config("configs/autoregressive/nano.yaml")
    cfg.set_context_window(16)
    cfg.model.d_model, cfg.model.n_heads, cfg.model.n_layers, cfg.model.d_ff = 32, 4, 2, 64
    torch.manual_seed(0)
    model = get_variant(cfg.variant)(vocab_size=50, cfg=cfg)
    old = {k.replace("backbone.layers.", "backbone.stack.layers.").replace("backbone.norm.", "backbone.stack.norm."): v
           for k, v in model.state_dict().items()}
    torch.manual_seed(5)
    other = get_variant(cfg.variant)(vocab_size=50, cfg=cfg)
    other.load_state_dict(old)
    ids = torch.randint(0, 50, (1, 16))
    batch = {"input_ids": ids, "attention_mask": torch.ones_like(ids)}
    assert torch.equal(other.eval()(batch), model.eval()(batch))


def test_the_parameter_order_is_unchanged_so_an_old_optimizer_state_still_lines_up():
    """Optimizer state is saved by parameter *position*, so the order must stay: embedding, blocks, final norm."""
    names = [name for name, _ in _backbone().named_parameters()]
    groups = [name.split(".")[0] for name in names]
    assert groups[0] == "token_emb" and groups[-1] == "norm"
    assert groups == sorted(groups, key=["token_emb", "layers", "norm"].index)  # never interleaved
    assert "lm_head.weight" not in names  # tied to the embedding, so it is not a second parameter


def test_the_backbone_has_the_blocks_and_the_final_norm_directly():
    backbone = _backbone()
    assert len(backbone.layers) == 3 and all(isinstance(block, TransformerBlock) for block in backbone.layers)
    assert not hasattr(backbone, "stack") and backbone.norm.weight.shape == (32,)


# ----- the single auxiliary-loss function -----
def test_a_dense_model_has_no_auxiliary_loss():
    backbone = _backbone()
    backbone(torch.randint(0, 50, (2, 16)))
    assert float(auxiliary_loss(backbone)) == 0.0


def test_the_auxiliary_loss_is_the_sum_over_every_moe_layer_and_only_while_training():
    backbone = _backbone(ffn_type="moe").train()
    backbone(torch.randint(0, 50, (4, 16)))
    layer_losses = [b.ffn.aux_loss for b in backbone.layers]
    assert len(layer_losses) == 3 and all(float(x) > 0 for x in layer_losses)
    assert torch.allclose(auxiliary_loss(backbone), torch.stack(layer_losses).sum())
    assert torch.allclose(auxiliary_loss(backbone.layers[1]), layer_losses[1])  # works on a single block too
    backbone.eval()(torch.randint(0, 50, (4, 16)))
    assert float(auxiliary_loss(backbone)) == 0.0  # the layers only compute it in training


def test_the_auxiliary_loss_before_any_forward_pass_is_zero_and_a_parameterless_module_is_fine():
    assert float(auxiliary_loss(_backbone(ffn_type="moe"))) == 0.0  # aux_loss is still None in every layer
    assert float(auxiliary_loss(torch.nn.Identity())) == 0.0


def test_the_auxiliary_loss_has_gradients_back_to_the_router():
    backbone = _backbone(ffn_type="moe").train()
    backbone(torch.randint(0, 50, (4, 16)))
    auxiliary_loss(backbone).backward()
    assert backbone.layers[0].ffn.router.gate.weight.grad.abs().sum() > 0


# ----- the shared block builder -----
def test_build_blocks_makes_independent_blocks_that_share_the_positional_encoding():
    cfg = _cfg(ffn_type="moe")
    positional = build_positional_encoding("rope", head_dim=8)
    blocks = build_blocks(cfg, 3, positional)
    assert len(blocks) == 3 and all(isinstance(b.ffn, DeepseekMoE) for b in blocks)
    assert blocks[0].attn.proj.weight is not blocks[1].attn.proj.weight  # independent weights
    assert blocks[0].attn.heads[0].positional is positional is blocks[2].attn.heads[0].positional  # one shared encoding
    assert len(build_blocks(cfg, 0, positional)) == 0
