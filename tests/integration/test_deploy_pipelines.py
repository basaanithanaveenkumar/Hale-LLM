"""Real deployment runs of every variant (tiny models): ONNX Runtime everywhere, MLX on Apple
Silicon.
"""

from __future__ import annotations

import json

import pytest
from hale_core.deploy import DeployRunner

from hale_llm.deploy.adapters.run_config_resolver import REPO_ROOT

pytestmark = pytest.mark.integration
CI = REPO_ROOT / "configs/deploy/ci"


@pytest.mark.parametrize("path", sorted(CI.glob("*_onnx_ort.yaml")), ids=lambda p: p.stem)
def test_onnx_runtime_parity(path, tmp_path) -> None:
    report = DeployRunner.from_file(path, work_dir=tmp_path).run()
    assert report.ok, [(r.name, r.message) for r in report.results]
    run = next(tmp_path.iterdir())
    parity = json.loads((run / "onnx_parity/parity.json").read_text())
    assert set(parity) == {"min", "opt", "max"}
    assert all(
        m["ok"] and m["max_abs"] < 1e-3 for corner in parity.values() for m in corner.values()
    )


@pytest.mark.parametrize("path", sorted(CI.glob("*_mlx_fp32.yaml")), ids=lambda p: p.stem)
def test_mlx_parity(path, tmp_path) -> None:
    pytest.importorskip("mlx.core")
    report = DeployRunner.from_file(path, work_dir=tmp_path).run()
    assert report.ok, [(r.name, r.message) for r in report.results]
    run = next(tmp_path.iterdir())
    parity = json.loads((run / "mlx_function_parity/parity.json").read_text())
    assert all(m["max_abs"] < 1e-4 for corner in parity.values() for m in corner.values())
    assert (run / "MLXConvertStage/model.mlxfn").exists()
