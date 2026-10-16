"""Deployment adapters of every variant: registry scope, signatures, forwards, calibration, errors."""

from __future__ import annotations

import pytest
import torch
import yaml

import hale_llm.deploy  # noqa: F401 - registers the adapters
from hale_core.deploy.calibration import AdapterCalibrationSource
from hale_core.deploy.exceptions import ConfigError, RegistryError
from hale_core.deploy.registry import ADAPTERS
from hale_llm.deploy import (
    LLM_ADAPTERS,
    AutoregressiveLMAdapter,
    LoopedLMAdapter,
    MaskedDiffusionLMAdapter,
)
from hale_llm.deploy.adapters.calibration_corpus import CalibrationCorpus
from hale_llm.deploy.adapters.run_config_resolver import REPO_ROOT, RunConfigResolver

TINY_DIR = REPO_ROOT / "configs/deploy/_shared/tiny"
VARIANTS = sorted(p.stem.removesuffix("_adapter") for p in TINY_DIR.glob("*_adapter.yaml"))

#: variant -> (input names, output names, static seq)
CONTRACTS = {
    "autoregressive": (["input_ids"], ("logits",), False),
    "looped": (["input_ids"], ("logits",), False),
    "mtp": (["input_ids"], ("logits", "draft_logits_1", "draft_logits_2"), False),
    "diffusion": (["x_t", "t"], ("logits",), True),
    "flow_matching": (["x_t", "t"], ("logits",), True),
    "block_diffusion": (["x_clean", "x_noised", "t_noised"], ("logits",), True),
    "block_flow_matching": (["x_clean", "x_t", "t"], ("logits",), True),
}


def tiny(variant: str):
    cfg = yaml.safe_load((TINY_DIR / f"{variant}_adapter.yaml").read_text())["adapter"]
    return ADAPTERS.build(cfg)


def test_every_variant_has_a_scoped_adapter() -> None:
    assert VARIANTS == sorted(CONTRACTS)
    assert len(LLM_ADAPTERS) == 7
    assert ADAPTERS.get("hale_llm.AutoregressiveLMAdapter") is AutoregressiveLMAdapter
    assert LLM_ADAPTERS.get("TorchModuleAdapter") is ADAPTERS.get("TorchModuleAdapter")  # parent fallback
    with pytest.raises(RegistryError):
        ADAPTERS.get("AutoregressiveLMAdapter")  # only reachable through its scope


@pytest.mark.parametrize("variant", sorted(CONTRACTS))
def test_signature_forward_and_calibration_contract(variant: str) -> None:
    adapter = tiny(variant)
    names, outputs, static = CONTRACTS[variant]
    sig = adapter.signature()
    assert sig.input_names == names and sig.outputs == outputs
    assert sig.dims["seq"].max == 32 and sig.dims["seq"].is_dynamic is not static
    assert all(s.value_range == (0, 128) for s in sig.inputs if s.dtype == "int64")
    module = adapter.export_module(adapter.load())
    args = sig.as_args(sig.sample("trace"))
    out = module(*args)
    assert len(out) == len(outputs) and all(o.shape == (2, sig.dims["seq"].trace, 128) for o in out)
    batch = adapter.calibration_inputs(torch.randint(0, 127, (1, sig.dims["seq"].opt)))
    assert list(batch) == names
    for spec in sig.inputs:
        assert batch[spec.name].dtype == spec.torch_dtype
        if spec.dtype == "float32":
            assert 0.0 <= float(batch[spec.name].min()) and float(batch[spec.name].max()) <= 1.0


def test_vocab_and_weights_come_from_a_trainer_checkpoint(tmp_path) -> None:
    source = tiny("autoregressive")
    model = source.load()
    ckpt = tmp_path / "last.pt"
    torch.save({"model_state_dict": model.state_dict(), "vocab_size": 128, "variant": "autoregressive"}, ckpt)
    cfg = yaml.safe_load((TINY_DIR / "autoregressive_adapter.yaml").read_text())["adapter"]
    cfg.pop("vocab_size")
    adapter = ADAPTERS.build({**cfg, "checkpoint": str(ckpt)})
    assert adapter.vocab_size == 128
    loaded = adapter.load()
    assert all(torch.equal(a, b) for a, b in zip(loaded.state_dict().values(), model.state_dict().values(), strict=True))


def test_config_errors_are_explicit() -> None:
    with pytest.raises(ConfigError, match="not found"):
        RunConfigResolver("configs/nope.yaml")
    with pytest.raises(ConfigError, match="deploys variant"):
        _ = MaskedDiffusionLMAdapter("configs/autoregressive/nano.yaml", vocab_size=8).run_config
    with pytest.raises(ConfigError, match="invalid overrides"):
        _ = AutoregressiveLMAdapter("configs/autoregressive/nano.yaml", vocab_size=8,
                                    overrides={"model": {"d_model": "wide"}}).run_config
    act = LoopedLMAdapter("configs/looped/act.yaml", vocab_size=8,
                          overrides={"model": {"d_model": 32, "n_heads": 2, "d_ff": 64}})
    with pytest.raises(ConfigError, match="ACT halting"):
        act.build_model()


def test_real_text_calibration_when_the_tokenizer_is_cached() -> None:
    if CalibrationCorpus("gpt2").tokenizer() is None:
        pytest.skip("gpt2 tokenizer not cached locally")
    adapter = tiny("block_diffusion")
    adapter._vocab_size = 50258  # real GPT-2 ids, tiny model is never built here
    source = AdapterCalibrationSource(adapter, num_batches=3)
    batches = list(source.batches(adapter.signature()))
    assert len(batches) == 3 and batches[0]["x_clean"].shape == (1, 32)
    assert not torch.equal(batches[0]["x_clean"], batches[1]["x_clean"])  # different text windows
