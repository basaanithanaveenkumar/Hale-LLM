"""Jarvislabs integration, tested with a fake SDK client and a fake ``jl`` runner (no network)."""

from __future__ import annotations

import subprocess
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from hale_llm.cloud.jarvislabs.errors import CloudError
from hale_llm.cloud.jarvislabs.instance_manager import InstanceManager
from hale_llm.cloud.jarvislabs.jl_cli import JlCli
from hale_llm.cloud.jarvislabs.project_archive import ProjectArchive
from hale_llm.cloud.jarvislabs.remote_training_job import RemoteTrainingJob
from hale_llm.cloud.jarvislabs.settings import JarvisSettings


class FakeInstances:
    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def create(self, **kwargs):
        self.calls.append(("create", kwargs))
        return SimpleNamespace(machine_id=42)

    def pause(self, machine_id):
        self.calls.append(("pause", machine_id))

    def destroy(self, machine_id):
        self.calls.append(("destroy", machine_id))


class FakeJl:
    """Records every command instead of running it."""

    def __init__(self, exec_output: str = "") -> None:
        self.commands: list[tuple] = []
        self.exec_output = exec_output

    def exec(self, machine_id, script, *, capture=False):
        self.commands.append(("exec", machine_id, script))
        return self.exec_output

    def upload(self, machine_id, source, destination):
        self.commands.append(("upload", machine_id, Path(source).name, destination))

    def download(self, machine_id, source, destination):
        self.commands.append(("download", machine_id, source, destination))


def test_settings_reject_unknown_gpu_and_region(tmp_path):
    with pytest.raises(ValueError):
        JarvisSettings(gpu_type="V100")
    with pytest.raises(ValueError):
        JarvisSettings(region="EU1")


def test_repo_settings_file_loads():
    settings = JarvisSettings.from_yaml("configs/cloud/jarvislabs.yaml")
    assert settings.region == "IN2" and settings.gpu_type == "A100"


def test_create_passes_settings_and_a_valid_name():
    client = SimpleNamespace(instances=FakeInstances())
    manager = InstanceManager(JarvisSettings(gpu_type="H100", num_gpus=2), client=client)
    instance = manager.create("autoregressive_moe/nano")
    (_, kwargs), = client.instances.calls
    assert instance.machine_id == 42
    assert kwargs["gpu_type"] == "H100" and kwargs["num_gpus"] == 2 and kwargs["region"] == "IN2"
    assert kwargs["name"] == "hale-autoregressive_moe-nano" and len(kwargs["name"]) <= 40


def test_instance_name_is_limited_to_40_safe_characters():
    name = InstanceManager.instance_name("hale", "a" * 80 + "!?")
    assert len(name) == 40 and set(name) <= set("abcdefghijklmnopqrstuvwxyz-")


def test_pause_and_destroy_reach_the_sdk():
    client = SimpleNamespace(instances=FakeInstances())
    manager = InstanceManager(JarvisSettings(), client=client)
    manager.pause(7)
    manager.destroy(7)
    assert client.instances.calls == [("pause", 7), ("destroy", 7)]


def test_archive_holds_tracked_files_only(tmp_path):
    repo = tmp_path / "repo"
    (repo / "ext_pkgs").mkdir(parents=True)
    (repo / "keep.py").write_text("x = 1\n")
    (repo / "ext_pkgs" / "big.py").write_text("skip me\n")
    (repo / "untracked.txt").write_text("not tracked\n")
    for cmd in (["git", "init", "-q"], ["git", "add", "keep.py", "ext_pkgs/big.py"]):
        subprocess.run(cmd, cwd=repo, check=True)
    archive = ProjectArchive(repo).build(tmp_path / "src.tgz")
    with tarfile.open(archive) as tar:
        assert tar.getnames() == ["keep.py"]


def test_deploy_uploads_then_installs(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "a.py").write_text("1\n")
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "add", "a.py"], cwd=repo, check=True)
    jl = FakeJl()
    RemoteTrainingJob(JarvisSettings(), jl, repo).deploy(5)
    kinds = [c[0] for c in jl.commands]
    assert kinds == ["upload", "exec"]
    assert jl.commands[0][2:] == ("hale-llm-src.tgz", "/home/")
    assert "uv sync" in jl.commands[1][2] and "/home/hale-llm" in jl.commands[1][2]


def test_start_runs_training_detached_with_quoted_args(tmp_path):
    jl = FakeJl()
    RemoteTrainingJob(JarvisSettings(), jl, tmp_path).start(5, ["--config", "configs/a b.yaml", "--steps", "10"])
    script = jl.commands[0][2]
    assert "setsid nohup sh -c" in script and "'configs/a b.yaml'" in script.replace("'\"'\"'", "'")
    assert "--steps 10" in script and script.rstrip().endswith("&")


def test_running_until_the_exit_file_appears(tmp_path):
    job = lambda out: RemoteTrainingJob(JarvisSettings(), FakeJl(out), tmp_path)
    assert job("").is_running(1) is True
    assert job("0\n").is_running(1) is False and job("1\n").exit_code(1) == 1


def test_start_records_the_exit_code(tmp_path):
    jl = FakeJl()
    RemoteTrainingJob(JarvisSettings(), jl, tmp_path).start(5, ["--config", "c.yaml"])
    assert "echo $? > train.exit" in jl.commands[0][2] and "rm -f train.exit" in jl.commands[0][2]


def test_wait_polls_until_training_ends(tmp_path, monkeypatch):
    answers = iter(["", "", "0\n"])

    class SequencedJl(FakeJl):
        def exec(self, machine_id, script, *, capture=False):
            return next(answers)

    monkeypatch.setattr("time.sleep", lambda s: None)
    assert RemoteTrainingJob(JarvisSettings(), SequencedJl(), tmp_path).wait_until_finished(1) == 0


def test_finish_pauses_even_when_the_download_fails(tmp_path):
    from hale_llm.cli.cloud import _finish

    class Job:
        def wait_until_finished(self, *a): return 0
        def fetch(self, *a): raise CloudError("download failed")

    paused = []
    manager = SimpleNamespace(pause=paused.append)
    with pytest.raises(CloudError):
        _finish(manager, Job(), 3, 0)
    assert paused == [3]


def test_fetch_downloads_the_experiments_folder(tmp_path):
    jl = FakeJl()
    where = RemoteTrainingJob(JarvisSettings(), jl, tmp_path).fetch(9, tmp_path / "out")
    assert where.is_dir()
    assert jl.commands == [("download", 9, "/home/hale-llm/data/experiments", str(tmp_path / "out"))]


def test_jl_cli_reports_failures(tmp_path, monkeypatch):
    fake = tmp_path / "jl"
    fake.write_text("#!/bin/sh\necho boom >&2\nexit 3\n")
    fake.chmod(0o755)
    monkeypatch.setenv("PATH", f"{tmp_path}:/usr/bin:/bin")
    with pytest.raises(CloudError, match="exit 3"):
        JlCli().exec(1, "true")


def test_jl_cli_missing_program_has_a_clear_error(monkeypatch):
    monkeypatch.setenv("PATH", "/nonexistent")
    with pytest.raises(CloudError, match="not found"):
        JlCli()


def test_launch_passes_unknown_flags_through_to_training():
    from hale_llm.cli.cloud import _build_parser

    args, extra = _build_parser().parse_known_args(["launch", "--config", "c.yaml", "--steps", "2000"])
    assert args.config == "c.yaml" and extra == ["--steps", "2000"]
