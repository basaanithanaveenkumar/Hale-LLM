"""Thin wrapper over the ``jl`` command line (exec, upload, download).

The Python SDK can create and pause instances but has no way to run commands or move files,
so those two jobs go through the ``jl`` program that ships with the ``jarvislabs`` package.
"""

from __future__ import annotations

import shutil
import subprocess

from loguru import logger

from hale_llm.cloud.jarvislabs.errors import CloudError


class JlCli:
    """Run ``jl`` subcommands and raise ``CloudError`` when one fails."""

    def __init__(self, executable: str = "jl") -> None:
        # Step 1: find the program once, so a missing install fails with a clear message.
        self.executable = shutil.which(executable)
        if self.executable is None:
            raise CloudError("`jl` not found. Install it with: uv sync --extra cloud, then run `jl setup`.")

    def _run(self, args: list[str], *, capture: bool = False) -> str:
        """Run one ``jl`` command; return stdout when ``capture`` is set."""
        command = [self.executable, *args]
        logger.debug("running: {}", " ".join(command))
        result = subprocess.run(command, text=True, capture_output=capture)
        if result.returncode != 0:
            detail = (result.stderr or "").strip()
            raise CloudError(f"`jl {args[0]}` failed (exit {result.returncode}). {detail}")
        return result.stdout if capture else ""

    def exec(self, machine_id: int, script: str, *, capture: bool = False) -> str:
        """Run a shell script on the instance (login shell, so pipes and ``cd`` work)."""
        return self._run(["exec", str(machine_id), "--", "sh", "-lc", script], capture=capture)

    def upload(self, machine_id: int, source: str, destination: str) -> None:
        """Copy a local file or folder to the instance."""
        self._run(["upload", str(machine_id), source, destination])

    def download(self, machine_id: int, source: str, destination: str) -> None:
        """Copy a remote folder to a local one."""
        self._run(["download", str(machine_id), source, destination, "-r"])
