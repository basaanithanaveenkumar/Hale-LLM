"""Ship the code to an instance, start training in the background, and follow it."""

from __future__ import annotations

import shlex
import tempfile
import time
from pathlib import Path

from loguru import logger

from hale_llm.cloud.jarvislabs.jl_cli import JlCli
from hale_llm.cloud.jarvislabs.project_archive import ProjectArchive
from hale_llm.cloud.jarvislabs.settings import JarvisSettings

_ARCHIVE_NAME = "hale-llm-src.tgz"
_EXIT_FILE = "train.exit"
_UPLOAD_DIR = "/home/"  # where `jl upload` puts a file on a container instance


class RemoteTrainingJob:
    """One training run on one instance. Training survives you closing the terminal."""

    def __init__(self, settings: JarvisSettings, jl: JlCli, repo_root: Path) -> None:
        self.settings = settings
        self.jl = jl
        self.repo_root = repo_root

    @property
    def log_path(self) -> str:
        """Remote file that receives the training output."""
        return f"{self.settings.remote_dir}/train.log"

    def deploy(self, machine_id: int) -> None:
        """Upload the code, unpack it, and install dependencies with uv."""
        remote_dir = shlex.quote(self.settings.remote_dir)
        with tempfile.TemporaryDirectory() as tmp:
            # Step 1: pack the tracked files and copy the archive to the instance.
            archive = ProjectArchive(self.repo_root).build(Path(tmp) / _ARCHIVE_NAME)
            self.jl.upload(machine_id, str(archive), _UPLOAD_DIR)
        # Step 2: unpack, install uv if the image lacks it, and install the project.
        self.jl.exec(
            machine_id,
            "set -e; export PATH=$HOME/.local/bin:$PATH; "
            "command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh; "
            f"mkdir -p {remote_dir} && tar -xzf {_UPLOAD_DIR}{_ARCHIVE_NAME} -C {remote_dir} && "
            f"cd {remote_dir} && uv sync",
        )

    def start(self, machine_id: int, train_args: list[str]) -> None:
        """Start ``hale-llm-train`` detached. Output goes to ``train.log``, the exit code to ``train.exit``."""
        remote_dir = shlex.quote(self.settings.remote_dir)
        train = shlex.join(["uv", "run", "hale-llm-train", *train_args])
        wrapped = shlex.quote(f"{train}; echo $? > {_EXIT_FILE}")  # the file appears only when training ends
        self.jl.exec(
            machine_id,
            "export PATH=$HOME/.local/bin:$PATH; "
            f"cd {remote_dir} && rm -f {_EXIT_FILE} && setsid nohup sh -c {wrapped} > train.log 2>&1 < /dev/null &",
        )
        logger.info("training started on {}; follow it with: hale-llm-cloud logs {}", machine_id, machine_id)

    def logs(self, machine_id: int, tail: int = 40) -> str:
        """Last lines of the training log."""
        return self.jl.exec(machine_id, f"tail -n {int(tail)} {shlex.quote(self.log_path)}", capture=True)

    def exit_code(self, machine_id: int) -> int | None:
        """Exit code of the finished training run, or None while it is still running."""
        out = self.jl.exec(
            machine_id, f"cat {shlex.quote(self.settings.remote_dir)}/{_EXIT_FILE} 2>/dev/null || true", capture=True
        ).strip()
        return int(out) if out else None

    def is_running(self, machine_id: int) -> bool:
        """True until the training process has written its exit code."""
        return self.exit_code(machine_id) is None

    def wait_until_finished(self, machine_id: int, poll_seconds: float = 60.0) -> int:
        """Block until training ends (success or crash) and return its exit code."""
        while (code := self.exit_code(machine_id)) is None:
            time.sleep(poll_seconds)
        logger.info("training on {} finished with exit code {}", machine_id, code)
        return code

    def fetch(self, machine_id: int, destination: Path) -> Path:
        """Copy the remote ``data/experiments`` folder (checkpoints, TensorBoard) to ``destination``."""
        destination.mkdir(parents=True, exist_ok=True)
        self.jl.download(machine_id, f"{self.settings.remote_dir}/data/experiments", str(destination))
        return destination
