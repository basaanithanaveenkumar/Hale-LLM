"""Build the roadmap notebooks from their Markdown sources and execute them.

Why Markdown sources?
---------------------
``.ipynb`` files are JSON: painful to review in a diff and easy to corrupt by hand. Every notebook is
therefore authored as ``source/NN_topic.md``:

* fenced ```` ```python ```` blocks become **code cells**;
* everything else becomes **markdown cells** (including ```` ```mermaid ```` diagrams, which JupyterLab
  >= 4.1, VS Code and GitHub render natively);
* ```` ```py ```` blocks stay inside markdown as *display-only* code (pseudo-code / cheat sheets);
* a line containing only ``+++`` splits a long markdown stretch into separate cells.

The script executes each notebook top-to-bottom (cwd = ``notebooks/rl/``) so the committed ``.ipynb``
files contain every plot, GIF and printed trace - a reader can follow them on GitHub without
installing anything. Execution doubles as the test suite: the ``assert`` statements inside the
notebooks check that each algorithm actually learns, so a broken algorithm fails the build.

Usage::

    .venv/bin/python notebooks/rl/build_notebooks.py               # build + execute all
    .venv/bin/python notebooks/rl/build_notebooks.py 03 07         # only sources starting with 03 / 07
    .venv/bin/python notebooks/rl/build_notebooks.py --no-execute  # just convert (fast, no outputs)
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path
from typing import List

import nbformat
from jupyter_client.kernelspec import KernelSpec
from jupyter_client.manager import KernelManager
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

ROOT = Path(__file__).resolve().parent          # notebooks/rl
SOURCE_DIR = ROOT / "source"
NOTEBOOK_DIR = ROOT                              # built notebooks sit next to this script

# Opening fence must be exactly ```python at the start of a line; the closing fence is a bare ```.
CODE_BLOCK = re.compile(r"^```python[ \t]*\n(.*?)^```[ \t]*$", re.MULTILINE | re.DOTALL)

# Every notebook starts with this hidden-ish setup cell so `import rlkit` works from notebooks/.
PRELUDE = (
    "# Setup: make the shared `rlkit` helpers (plots, GIFs, toy envs, tiny models) importable.\n"
    "import sys, pathlib\n"
    "sys.path.insert(0, str(pathlib.Path.cwd()))\n"
    "%matplotlib inline"
)


def markdown_cells(text: str) -> List[nbformat.NotebookNode]:
    """Split a markdown stretch on ``+++`` lines, dropping empty pieces."""
    cells = []
    for chunk in re.split(r"^\+\+\+[ \t]*$", text, flags=re.MULTILINE):
        chunk = chunk.strip("\n")
        if chunk.strip():
            cells.append(nbformat.v4.new_markdown_cell(chunk))
    return cells


def parse_source(text: str) -> nbformat.NotebookNode:
    """Convert one Markdown source file into an (unexecuted) notebook."""
    nb = nbformat.v4.new_notebook()
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    nb.metadata["language_info"] = {"name": "python"}
    cursor = 0
    first_code = True
    for match in CODE_BLOCK.finditer(text):
        nb.cells.extend(markdown_cells(text[cursor:match.start()]))
        if first_code:
            nb.cells.append(nbformat.v4.new_code_cell(PRELUDE))
            first_code = False
        nb.cells.append(nbformat.v4.new_code_cell(match.group(1).rstrip("\n")))
        cursor = match.end()
    nb.cells.extend(markdown_cells(text[cursor:]))
    return nb


def interpreter_kernel() -> KernelManager:
    """A kernel that runs on *this* interpreter.

    The notebook metadata says ``python3`` (so it opens anywhere), but a user-level ``python3``
    kernelspec can point at some other, possibly deleted, environment. Pinning the kernel to
    ``sys.executable`` makes the build use exactly the environment the script was launched from.
    """
    km = KernelManager(kernel_name="python3")
    km._kernel_spec = KernelSpec(argv=[sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
                                 display_name="Python 3", language="python")
    return km


def build(source: Path, execute: bool, timeout: int) -> float:
    """Build one notebook; returns the seconds it took. Raises on any cell error."""
    start = time.perf_counter()
    nb = parse_source(source.read_text(encoding="utf-8"))
    if execute:
        client = NotebookClient(nb, km=interpreter_kernel(), timeout=timeout,
                                resources={"metadata": {"path": str(NOTEBOOK_DIR)}})
        client.execute()
    out = NOTEBOOK_DIR / f"{source.stem}.ipynb"
    nbformat.write(nb, out)
    return time.perf_counter() - start


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("prefixes", nargs="*", help="only build sources whose filename starts with these")
    parser.add_argument("--no-execute", action="store_true", help="convert without running the cells")
    parser.add_argument("--timeout", type=int, default=900, help="per-cell timeout in seconds")
    args = parser.parse_args(argv)

    NOTEBOOK_DIR.mkdir(exist_ok=True)
    sources = sorted(SOURCE_DIR.glob("*.md"))
    if args.prefixes:
        sources = [s for s in sources if any(s.name.startswith(p) for p in args.prefixes)]
    if not sources:
        print("no matching sources", file=sys.stderr)
        return 1

    failures = []
    for src in sources:
        try:
            seconds = build(src, execute=not args.no_execute, timeout=args.timeout)
            print(f"  ok   {src.stem:<44} {seconds:6.1f}s")
        except CellExecutionError as err:
            failures.append(src.stem)
            # The first ~40 lines of the traceback are enough to locate the failing cell.
            print(f"  FAIL {src.stem}\n" + "\n".join(str(err).splitlines()[:40]), file=sys.stderr)
    print(f"\n{len(sources) - len(failures)}/{len(sources)} notebooks built")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
