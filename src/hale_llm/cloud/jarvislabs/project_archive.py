"""Pack the project files (everything git does not ignore) into one tarball to ship to the instance."""

from __future__ import annotations

import subprocess
import tarfile
from pathlib import Path

from loguru import logger

# Folders that are never needed to train: submodules, demo assets, the report and page, and results
# already downloaded from earlier cloud runs.
_SKIPPED_PREFIXES = ("ext_pkgs/", "assets/", "technical_report/", "project-page/", "data/cloud_runs/")


class ProjectArchive:
    """Build a ``.tgz`` of the project files.

    Tracked files and new files you have not committed yet are both included (a new config you
    forgot to ``git add`` must still reach the instance); files in ``.gitignore`` are not.
    """

    def __init__(self, repo_root: Path) -> None:
        self.repo_root = repo_root

    def tracked_files(self) -> list[Path]:
        """Tracked plus untracked-not-ignored files that exist on disk, minus the skipped folders."""
        out = subprocess.run(
            ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"], cwd=self.repo_root, capture_output=True, text=True, check=True
        ).stdout
        names = [n for n in out.split("\0") if n and not n.startswith(_SKIPPED_PREFIXES)]
        return [Path(n) for n in names if (self.repo_root / n).is_file()]

    def build(self, target: Path) -> Path:
        """Write the archive to ``target`` and return it."""
        files = self.tracked_files()
        with tarfile.open(target, "w:gz") as tar:
            for rel in files:
                tar.add(self.repo_root / rel, arcname=str(rel))
        logger.info("packed {} files into {} ({:.1f} MB)", len(files), target.name, target.stat().st_size / 1e6)
        return target
