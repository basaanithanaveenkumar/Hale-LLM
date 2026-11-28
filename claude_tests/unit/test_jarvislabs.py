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

    def get(self, machine_id):
        self.calls.append(("get", machine_id))
        return next(self.states)

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
    assert settings.region in ("IN1", "IN2", None) and settings.num_gpus >= 1


def test_create_passes_settings_and_a_valid_name():
    client = SimpleNamespace(instances=FakeInstances())
    manager = InstanceManager(JarvisSettings(gpu_type="H100", num_gpus=2), client=client)
    instance = manager.create("autoregressive_moe/nano")
    (_, kwargs), = client.instances.calls
    assert instance.machine_id == 42
    assert kwargs["gpu_type"] == "H100" and kwargs["num_gpus"] == 2 and kwargs["region"] == "IN2"
    assert kwargs["name"] == "hale-autoregressive_moe-nano" and len(kwargs["name"]) <= 40


def test_spot_flag_reaches_the_sdk():
    client = SimpleNamespace(instances=FakeInstances())
    InstanceManager(JarvisSettings(spot=True), client=client).create("x")
    InstanceManager(JarvisSettings(), client=client).create("x")
    assert [c[1]["is_spot"] for c in client.instances.calls] == [True, False]


def test_waits_until_ssh_is_available(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    client = SimpleNamespace(instances=FakeInstances())
    client.instances.states = iter([
        SimpleNamespace(status="Running", ssh_command=None),  # the case that broke the first launch
        SimpleNamespace(status="Running", ssh_command="ssh -p 1 host"),
    ])
    instance = InstanceManager(JarvisSettings(), client=client).wait_until_ssh_ready(9)
    assert instance.ssh_command == "ssh -p 1 host" and len(client.instances.calls) == 2


def test_gives_up_if_ssh_never_appears(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    client = SimpleNamespace(instances=FakeInstances())
    client.instances.states = iter(SimpleNamespace(status="Running", ssh_command=None) for _ in range(1000))
    with pytest.raises(CloudError, match="not reachable"):
        InstanceManager(JarvisSettings(), client=client).wait_until_ssh_ready(9, timeout_seconds=0)


def test_missing_ssh_key_is_reported_before_anything_is_rented():
    client = SimpleNamespace(instances=FakeInstances(), ssh_keys=SimpleNamespace(list=lambda: []))
    with pytest.raises(CloudError, match="ssh-key add"):
        InstanceManager(JarvisSettings(), client=client).ensure_ssh_key()
    assert client.instances.calls == []  # nothing was created
    client.ssh_keys = SimpleNamespace(list=lambda: [object()])
    InstanceManager(JarvisSettings(), client=client).ensure_ssh_key()  # a registered key passes


def test_instance_name_is_limited_to_40_safe_characters():
    name = InstanceManager.instance_name("hale", "a" * 80 + "!?")
    assert len(name) == 40 and set(name) <= set("abcdefghijklmnopqrstuvwxyz-")


def test_pause_and_destroy_reach_the_sdk():
    client = SimpleNamespace(instances=FakeInstances())
    manager = InstanceManager(JarvisSettings(), client=client)
    manager.pause(7)
    manager.destroy(7)
    assert client.instances.calls == [("pause", 7), ("destroy", 7)]


def test_archive_holds_new_and_tracked_files_but_not_ignored_or_skipped_ones(tmp_path):
    repo = tmp_path / "repo"
    for folder in ("ext_pkgs", "configs", "data/cloud_runs"):
        (repo / folder).mkdir(parents=True)
    (repo / "keep.py").write_text("x = 1\n")
    (repo / "ext_pkgs" / "big.py").write_text("skip me\n")
    (repo / "data" / "cloud_runs" / "old.txt").write_text("old results\n")
    (repo / "configs" / "new.yaml").write_text("not committed or added yet\n")  # the bug: this used to be left behind
    (repo / "secret.pt").write_text("ignored\n")
    (repo / ".gitignore").write_text("*.pt\n")
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "add", "keep.py", "ext_pkgs/big.py"], cwd=repo, check=True)
    archive = ProjectArchive(repo).build(tmp_path / "src.tgz")
    with tarfile.open(archive) as tar:
        assert sorted(tar.getnames()) == [".gitignore", "configs/new.yaml", "keep.py"]


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
    manager = SimpleNamespace(pause_if_running=paused.append)
    with pytest.raises(CloudError):
        _finish(manager, Job(), 3, 0)
    assert paused == [3]


def test_fetch_downloads_the_experiments_folder(tmp_path):
    jl = FakeJl("yes\n")
    where = RemoteTrainingJob(JarvisSettings(), jl, tmp_path).fetch(9, tmp_path / "out")
    assert where.is_dir()
    assert jl.commands[-1] == ("download", 9, "/home/hale-llm/data/experiments", str(tmp_path / "out"))


def test_fetch_skips_the_download_when_there_are_no_results(tmp_path):
    jl = FakeJl("no\n")  # e.g. a dry run, or training that crashed at startup
    assert RemoteTrainingJob(JarvisSettings(), jl, tmp_path).fetch(9, tmp_path / "out") is None
    assert all(c[0] != "download" for c in jl.commands)


def test_a_failed_training_shows_the_log_and_still_pauses(tmp_path, capsys):
    from hale_llm.cli.cloud import _finish

    class Job:
        def wait_until_finished(self, *a): return 1
        def logs(self, *a): return "FileNotFoundError: configs/x.yaml"
        def fetch(self, *a): raise AssertionError("must not download after a failed run")

    paused = []
    with pytest.raises(CloudError, match="exit code 1"):
        _finish(SimpleNamespace(pause_if_running=paused.append), Job(), 4, 0)
    assert paused == [4] and "FileNotFoundError" in capsys.readouterr().out


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


def test_cloud_training_always_targets_cuda_unless_told_otherwise():
    from hale_llm.cli.cloud import _train_args

    assert _train_args("c.yaml", ["--steps", "5"]) == ["--config", "c.yaml", "--steps", "5", "--device", "cuda"]
    assert _train_args("c.yaml", ["--device", "cpu"]) == ["--config", "c.yaml", "--device", "cpu"]
