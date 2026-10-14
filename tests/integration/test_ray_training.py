"""hale-llm training on Ray Train: 2 CPU workers with DDP, checkpoint reported back."""

import sys
from pathlib import Path

import pytest

ray = pytest.importorskip("ray")

from hale_llm.config.schema import load_config  # noqa: E402
from hale_llm.training.trainer import train  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
SUPPORT = REPO / "tests" / "ray_support"
sys.path.insert(0, str(SUPPORT.parent))
from ray_support.tiny_tokenizer import TinyTokenizer  # noqa: E402


@pytest.fixture(scope="module")
def local_ray():
    ray.init(
        num_cpus=4,
        include_dashboard=False,
        log_to_driver=False,
        runtime_env={"py_modules": [str(SUPPORT)]},
    )
    yield
    ray.shutdown()


@pytest.mark.integration
def test_autoregressive_overfit_on_two_ray_workers(local_ray, tmp_path):
    cfg = load_config(REPO / "configs" / "autoregressive" / "small.yaml")
    cfg.train.steps = 8
    cfg.train.epochs = None
    cfg.train.batch_size = 4
    cfg.train.checkpoint_path = str(tmp_path / "ckpt" / "last.pt")
    cfg.train.resume = False
    cfg.model.d_model, cfg.model.n_heads, cfg.model.n_layers, cfg.model.d_ff = 32, 4, 1, 64
    cfg.model.max_length = 16
    cfg.data.overfit_text = "hello world hello world"
    cfg.data.n_overfit_copies = 16
    cfg.device = "cpu"
    cfg.logging.tensorboard = False
    cfg.logging.log_file = None
    cfg.logging.level = "WARNING"
    cfg.eval.every_n_epochs = None
    cfg.viz.enabled = False
    cfg.experiment.enabled = False
    cfg.train.ray = cfg.train.ray.model_copy(
        update={"enabled": True, "num_workers": 2, "storage_path": str(tmp_path / "ray")}
    )

    result = train(cfg, TinyTokenizer())

    assert result.ok, result.error
    assert result.metrics["global_step"] > 0
    assert result.checkpoint_path is not None
    assert any(Path(result.checkpoint_path).iterdir())
