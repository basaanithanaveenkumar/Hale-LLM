"""autoregressive_moe_mtp: a mixture-of-experts trunk with MTP depths, the model behind configs/autoregressive_mtp/nano.yaml."""

from __future__ import annotations

import pytest
import torch

import hale_llm.models  # noqa: F401
from hale_llm.config.run import RunConfig
from hale_llm.config.schema import load_config
from hale_llm.config.sections.model import ModelConfig
from hale_llm.dataloader.collate import get_collate
from hale_llm.inference import load_session
from hale_llm.metrics.llm.variants import AUTOREGRESSIVE_VARIANTS
from hale_llm.models.models_autoregressive.sample import sample_autoregressive
from hale_llm.models.models_mtp.speculative import SpeculativeDecoder
from hale_llm.nn.moe import auxiliary_loss
from hale_llm.registry import get_loss, get_sampler, get_variant
from hale_llm.training.lean_trainer import train_lean

VARIANT = "autoregressive_moe_mtp"


def _small(cfg, window=16):
    cfg.set_context_window(window)
    cfg.model.d_model, cfg.model.n_heads, cfg.model.n_layers, cfg.model.d_ff = 32, 4, 2, 48
    cfg.model.dropout, cfg.model.moe_router = 0.0, "topk"
    return cfg


def test_nano_yaml_is_the_moe_mtp_model_with_the_tuned_settings():
    cfg = load_config("configs/autoregressive_mtp/nano.yaml")
    assert cfg.variant == VARIANT and cfg.model.ffn_type == "moe" and cfg.model.mtp_block_config().ffn_type == "moe"
    assert (cfg.model.mtp_loss_weight, cfg.model.mtp_loss_weight_final, cfg.model.mtp_n_layers) == (0.7, 0.5, 2)


def test_the_variant_forces_a_mixture_of_experts_trunk_whatever_the_yaml_says():
    cfg = RunConfig(variant=VARIANT, model=ModelConfig(ffn_type="mlp"))
    assert cfg.model.ffn_type == "moe"
    cfg.model.ffn_type = "mlp"  # changed after validation: building the model must refuse
    with pytest.raises(ValueError, match="needs model.ffn_type: moe"):
        get_variant(VARIANT)(vocab_size=50, cfg=cfg)


def test_the_trunk_and_the_mtp_blocks_are_both_mixture_of_experts():
    cfg = _small(load_config("configs/autoregressive_mtp/nano.yaml"))
    torch.manual_seed(0)
    model = get_variant(VARIANT)(vocab_size=50, cfg=cfg).train()
    assert type(model.backbone.layers[0].ffn).__name__ == "DeepseekMoE" and model.uses_moe
    assert all(type(b.ffn).__name__ == "DeepseekMoE" for d in model.mtp.depths for b in d.blocks)
    ids = torch.randint(0, 50, (2, 16))
    labels = ids.clone()
    labels[:, :-1], labels[:, -1] = ids[:, 1:], -100
    loss = get_loss(VARIANT)(model, {"input_ids": ids, "attention_mask": torch.ones_like(ids), "labels": labels})
    trunk, mtp, both = auxiliary_loss(model.backbone), auxiliary_loss(model.mtp), auxiliary_loss(model)
    assert trunk > 0 and mtp > 0 and torch.allclose(both, trunk + mtp)  # the balancing loss of every expert layer counts
    loss.backward()
    assert all(p.grad is not None for p in model.parameters())


def test_it_is_registered_everywhere_a_variant_must_be():
    assert callable(get_loss(VARIANT)) and callable(get_sampler(VARIANT)) and callable(get_collate(VARIANT))
    assert VARIANT in AUTOREGRESSIVE_VARIANTS  # the accuracy / top-k metrics apply


def test_greedy_speculative_decoding_is_exact_for_the_moe_mtp_model():
    cfg = _small(load_config("configs/autoregressive_mtp/nano.yaml"), window=64)
    cfg.model.mtp_depth = 3
    torch.manual_seed(0)
    model = get_variant(VARIANT)(vocab_size=24, cfg=cfg).double().eval()
    model.backbone.token_emb.weight.data *= 4
    prompt = torch.randint(0, 24, (1, 5), generator=torch.Generator().manual_seed(1))
    no_eos = type("T", (), {"eos_token_id": None, "pad_token_id": None})()
    assert torch.equal(sample_autoregressive(model, no_eos, prompt, 30, temperature=0.0),
                       SpeculativeDecoder(model, no_eos, 0.0).generate(prompt, 30))


def test_train_save_load_and_generate_through_the_real_pipeline(tiny_tokenizer, tmp_path):
    cfg = _small(load_config("configs/autoregressive_mtp/nano.yaml"), window=32)
    cfg.train.steps, cfg.train.epochs, cfg.train.batch_size, cfg.train.lr, cfg.train.log_every = 20, None, 8, 3e-3, 5
    cfg.train.lean.accum_steps, cfg.train.lean.warmup_steps, cfg.train.lean.eval_every, cfg.train.lean.save_every = 2, 3, 10, 0
    cfg.train.checkpoint_path = str(tmp_path / "ckpt" / "last.pt")
    cfg.data.overfit_text, cfg.data.n_overfit_copies = "hello world hello world", 32
    cfg.device, cfg.experiment.enabled, cfg.train.resume = "cpu", False, False
    cfg.logging.tensorboard, cfg.logging.tensorboard_dir = True, str(tmp_path / "tb")
    cfg.logging.log_file, cfg.logging.level = None, "WARNING"
    losses = train_lean(cfg, tokenizer=tiny_tokenizer)
    assert losses[-1] < losses[0]

    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    acc = EventAccumulator(str(tmp_path / "tb"), size_guidance={"scalars": 0})
    acc.Reload()
    assert {round(e.value, 4) for e in acc.Scalars("train/mtp_weight")} == {0.7, 0.5}  # the tuned lambda schedule

    session = load_session(cfg, tokenizer=tiny_tokenizer, require_checkpoint=True)
    assert type(session.model).variant_name == VARIANT and len(session.model.mtp.depths[0].blocks) == 2
    session.cfg.sample.speculative = True
    fast = session.generate("hello", max_new_tokens=10, temperature=0.0)
    session.cfg.sample.speculative = False
    assert fast.token_ids.tolist() == session.generate("hello", max_new_tokens=10, temperature=0.0).token_ids.tolist()
