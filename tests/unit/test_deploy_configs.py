"""Every deployment config builds a valid pipeline (inheritance, adapters, stage arguments)."""

from __future__ import annotations

import pytest
from hale_core.deploy import DeployRunner

from hale_llm.deploy.adapters.run_config_resolver import REPO_ROOT
from hale_llm.deploy.deploy_cli import main

DEPLOY = REPO_ROOT / "configs/deploy"
CONFIGS = sorted(p for p in DEPLOY.rglob("*.yaml") if "_shared" not in p.parts)


def test_config_tree_is_complete() -> None:
    variants = {p.parent.name for p in CONFIGS if p.parent.name != "ci"}
    assert len(variants) == 7 and all(len(list((DEPLOY / v).glob("*.yaml"))) == 9 for v in variants)
    assert len(list((DEPLOY / "ci").glob("*.yaml"))) == 14


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: f"{p.parent.name}/{p.stem}")
def test_config_builds_a_valid_pipeline(path, tmp_path) -> None:
    runner = DeployRunner.from_file(path, work_dir=tmp_path)
    pipeline, ctx = runner.build()
    pipeline.validate()
    assert runner.spec.adapter["type"].startswith("hale_llm.")
    vendor_neutral = path.stem.endswith(
        "_onnx_ort"
    )  # CPU ONNX Runtime CI pipelines have no device profile
    assert vendor_neutral or ctx.device.get("vendor") in {"nvidia", "apple", "amd", "qualcomm"}


def test_cli_lists_the_scoped_adapters(capsys) -> None:
    assert main(["list"]) == 0
    out = capsys.readouterr().out
    assert "hale_llm.BlockFlowMatchingLMAdapter" in out and "TensorRTEngineStage" in out
