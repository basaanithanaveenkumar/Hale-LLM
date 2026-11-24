"""Device selection: cpu / cuda / rocm (AMD) / mps, with real training steps on the devices this machine has.

CUDA and ROCm are *simulated* (monkeypatched torch) when no such GPU is present - those tests check the
selection logic, not the hardware. The training tests run for real on cpu and on the best local accelerator.
"""

import argparse

import pytest
import torch

from hale_llm.utils import device as dev


@pytest.fixture
def fake_cuda(monkeypatch):
    """Pretend to be a 2-GPU NVIDIA box (no MPS)."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(torch.version, "hip", None, raising=False)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda i=0: f"Fake GPU {i}")
    monkeypatch.setattr(dev, "_mps_usable", lambda: False)


@pytest.fixture
def fake_rocm(fake_cuda, monkeypatch):
    """Pretend to be a ROCm box: torch reports the AMD GPU through the cuda namespace."""
    monkeypatch.setattr(torch.version, "hip", "6.2.0", raising=False)


@pytest.fixture
def cpu_only(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(dev, "_mps_usable", lambda: False)


def test_cpu_is_always_available(cpu_only):
    assert dev.get_device("cpu") == "cpu"
    assert dev.get_device(None) == "cpu" and dev.get_device("auto") == "cpu"
    assert dev.available_backends() == ["cpu"]


def test_cpu_only_requests_fall_back_not_crash(cpu_only):
    for req in ("cuda", "cuda:0", "rocm", "amd", "mps", "gpu"):
        assert dev.get_device(req) == "cpu", req


def test_strict_raises_instead_of_falling_back(cpu_only):
    for req in ("cuda", "rocm", "mps", "gpu"):
        with pytest.raises(dev.DeviceError):
            dev.resolve_device(req, strict=True)


def test_unknown_name_is_an_error():
    with pytest.raises(dev.DeviceError):
        dev.resolve_device("tpu")


def test_bad_cuda_index_is_an_error(fake_cuda):
    with pytest.raises(dev.DeviceError):
        dev.resolve_device("cuda:x")


def test_nvidia_cuda(fake_cuda):
    assert dev.detect_backend() == "cuda"
    assert dev.get_device() == "cuda" and dev.get_device("gpu") == "cuda"
    assert dev.get_device("cuda:1") == "cuda:1"
    assert dev.get_device("cuda:7") == "cuda"          # only 2 GPUs: falls back to the best, which is cuda
    assert dev.get_device("rocm") == "cuda"            # no HIP build -> not ROCm -> falls back to the best
    assert dev.dist_backend("cuda") == "nccl"


def test_amd_rocm_is_the_cuda_device_to_torch(fake_rocm):
    assert dev.detect_backend() == "rocm" and dev.available_backends()[0] == "rocm"
    for req in ("rocm", "amd", "hip", "cuda", "auto", "gpu"):
        assert dev.get_device(req) == "cuda", req      # torch names an AMD GPU "cuda"
    assert dev.resolve_device("amd", strict=True).type == "cuda"
    assert dev.dist_backend("rocm") == "nccl"          # RCCL answers to the name nccl
    assert dev.device_summary()["hip_version"] == "6.2.0"


def test_apple_mps(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(dev, "_mps_usable", lambda: True)
    assert dev.detect_backend() == "mps" and dev.get_device() == "mps"
    assert dev.get_device("mps") == "mps" and dev.get_device("gpu") == "mps"
    assert dev.dist_backend("mps") == "gloo"


def test_accepts_torch_device(cpu_only):
    assert dev.resolve_device(torch.device("cpu")) == torch.device("cpu")


def test_argparse_helper_resolves_at_parse_time(cpu_only):
    p = argparse.ArgumentParser()
    dev.add_device_argument(p)
    assert p.parse_args([]).device == "cpu"
    assert p.parse_args(["--device", "rocm"]).device == "cpu"      # unavailable -> falls back
    assert p.parse_args(["--device", "cpu"]).device == "cpu"


def test_helpers_are_safe_on_every_device():
    for name in ("cpu", "auto"):
        dev.synchronize(name)
        dev.empty_cache(name)
    dev.manual_seed_all(0)
    assert isinstance(dev.device_summary()["backends"], list)


def _local_devices():
    out = ["cpu"]
    if dev._mps_usable():
        out.append("mps")
    if torch.cuda.is_available():
        out.append("cuda")
    return out


@pytest.mark.parametrize("name", _local_devices())
def test_real_training_step_on_each_local_device(name):
    """A real forward / backward / AdamW step on the device: the loss is finite and the weights move."""
    device = torch.device(dev.get_device(name))
    assert device.type == name
    dev.manual_seed_all(0)
    model = torch.nn.Sequential(torch.nn.Linear(8, 16), torch.nn.GELU(), torch.nn.Linear(16, 1)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    x = torch.randn(32, 8, device=device)
    y = x.sum(1, keepdim=True)
    before = model[0].weight.detach().cpu().clone()
    losses = []
    for _ in range(40):
        loss = torch.nn.functional.mse_loss(model(x), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    dev.synchronize(device)
    assert all(map(lambda v: v == v and abs(v) < 1e6, losses))      # finite
    assert losses[-1] < 0.5 * losses[0]                              # it learned
    assert not torch.equal(before, model[0].weight.detach().cpu())  # the weights moved
    assert next(model.parameters()).device.type == name
