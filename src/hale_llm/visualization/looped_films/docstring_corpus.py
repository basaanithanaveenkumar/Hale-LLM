"""``docstring_corpus``: real English text that ships with the library - its own docstrings."""

from __future__ import annotations

import ast
import re
from pathlib import Path


def docstring_corpus(
    packages: tuple[str, ...] = ("hale_llm", "hale_core"), alphabet: str | None = None
) -> str:
    """Every module / class / function docstring of ``packages``, whitespace-normalised.

    The looped films and notebooks need a corpus where depth matters (the 4-sentence toy corpus
    is solved in one pass) without downloading anything. Docstrings are parsed with :mod:`ast`
    from the installed source files - nothing is imported, so optional dependencies are never
    touched. Characters outside ``alphabet`` are dropped.
    """
    import importlib

    chunks: list[str] = []
    for name in packages:
        try:
            root = Path(importlib.import_module(name).__file__).parent
        except ImportError:
            continue
        for path in sorted(root.rglob("*.py")):
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"))
            except (SyntaxError, UnicodeDecodeError):
                continue
            for node in ast.walk(tree):
                if isinstance(
                    node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef
                ):
                    doc = ast.get_docstring(node)
                    if doc and len(doc) > 40:
                        chunks.append(doc)
    text = re.sub(r"\s+", " ", " ".join(chunks))
    if alphabet is not None:
        allowed = set(alphabet)
        text = "".join(c for c in text if c in allowed)
    return text
