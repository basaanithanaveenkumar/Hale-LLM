"""Pack the git-tracked project files into one tarball to ship to the instance."""

from __future__ import annotations

import subprocess
import tarfile
from pathlib import Path

from loguru import logger

# Large folders that are tracked as submodules or demo assets and are not needed to train.
_SKIPPED_PREFIXES = ("ext_pkgs/", "assets/")


class ProjectArchive:
    """Build a ``.tgz`` of the tracked files (your working-tree edits included)."""

    def __init__(self, repo_root: Path) -> None:
        self.repo_root = repo_root

    def tracked_files(self) -> list[Path]:
        """Tracked files that exist on disk, minus the skipped folders."""
        out = subprocess.run(
            ["git", "ls-files", "-z"], cwd=self.repo_root, capture_output=True, text=True, check=True
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
