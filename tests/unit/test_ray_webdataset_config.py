"""The Ray and WebDataset sections are part of the run config and off by default."""

from hale_llm.config.schema import RunConfig


def test_defaults_are_off():
    cfg = RunConfig(variant="autoregressive")
    assert cfg.train.ray.enabled is False
    assert cfg.data.webdataset.enabled is False


def test_sections_parse_from_config_dicts():
    cfg = RunConfig.model_validate(
        {
            "variant": "diffusion",
            "train": {"ray": {"enabled": True, "num_workers": 4, "use_gpu": True}},
            "data": {"webdataset": {"enabled": True, "root": "/shards", "samples_per_shard": 500}},
        }
    )
    assert cfg.train.ray.num_workers == 4 and cfg.train.ray.resolved_backend() == "nccl"
    assert cfg.data.webdataset.root == "/shards" and cfg.data.webdataset.samples_per_shard == 500
