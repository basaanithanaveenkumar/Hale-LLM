"""make_dataloader with data.webdataset.enabled: same batches, lazy conversion, own shards."""

import pytest
import torch
from torch.utils.data import Dataset

pytest.importorskip("webdataset")

import hale_llm.dataloader.loaders as loaders  # noqa: E402
import hale_llm.models  # noqa: E402, F401  - registers the collates per variant
from hale_llm.config.schema import RunConfig  # noqa: E402


class FakeHFWindows(Dataset):
    """Stands in for HuggingFaceTextDataset (no download): fixed token windows per split."""

    built = 0

    def __init__(self, tokenizer, cfg, split: str) -> None:
        FakeHFWindows.built += 1
        g = torch.Generator().manual_seed(len(split))
        self.ids = torch.randint(2, 60, (13, cfg.model.max_length), generator=g)

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx):
        return self.ids[idx]


@pytest.fixture
def fake_hf(monkeypatch, tiny_tokenizer):
    monkeypatch.setattr(loaders, "HuggingFaceTextDataset", FakeHFWindows)
    monkeypatch.setattr(loaders, "setup_tokenizer", lambda name: tiny_tokenizer)
    FakeHFWindows.built = 0


def _cfg(tmp_path, enabled: bool, **data) -> RunConfig:
    return RunConfig(
        variant="autoregressive",
        model={"max_length": 16},
        train={"batch_size": 4},
        data={
            **data,
            "webdataset": {
                "enabled": enabled,
                "root": str(tmp_path / "shards"),
                "samples_per_shard": 5,
                "shuffle": False,
            },
        },
    )


def _flat(loader):
    return [{k: v.clone() for k, v in batch.items() if torch.is_tensor(v)} for batch in loader]


def test_validation_batches_are_identical_with_and_without_webdataset(fake_hf, tmp_path):
    plain = _flat(loaders.make_dataloader(_cfg(tmp_path, False), split="validation"))
    streamed = _flat(loaders.make_dataloader(_cfg(tmp_path, True), split="validation"))
    assert len(plain) == len(streamed) == 3  # 13 windows, batch 4, drop_last
    for a, b in zip(plain, streamed):
        assert a.keys() == b.keys()
        for key in a:
            assert torch.equal(a[key], b[key]), key


def test_dataset_is_only_built_when_shards_are_missing(fake_hf, tmp_path):
    loader = loaders.make_dataloader(_cfg(tmp_path, True), split="train")
    assert len(loader) == 3
    loaders.make_dataloader(_cfg(tmp_path, True), split="train")
    assert FakeHFWindows.built == 1  # second run streamed the existing shards


def test_each_data_config_gets_its_own_shards(fake_hf, tmp_path):
    loaders.make_dataloader(_cfg(tmp_path, True), split="train")
    loaders.make_dataloader(_cfg(tmp_path, True, stride_words=3), split="train")
    assert FakeHFWindows.built == 2
    assert len(list((tmp_path / "shards").iterdir())) == 2
