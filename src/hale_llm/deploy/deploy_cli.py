"""``hale-llm-deploy``: the HaleBlocks deployment CLI with this repo's adapters registered."""

from __future__ import annotations

from collections.abc import Sequence

from hale_core.deploy.cli.deploy_cli import DeployCLI


class HaleLLMDeployCLI(DeployCLI):
    """Thin subclass: importing :mod:`hale_llm.deploy` registers the ``hale_llm.*`` adapters
    before any subcommand runs (``list`` then shows them, ``run`` can build them even when a
    config forgets ``custom_imports``)."""

    def __init__(self) -> None:
        import hale_llm.deploy.adapters  # noqa: F401 - registration side effect

        super().__init__(prog="hale-llm-deploy")

    @staticmethod
    def cmd_list(args) -> int:
        """Root registries, plus the ``hale_llm`` scope (child scopes are not walked by the
        root).
        """
        from hale_llm.deploy.registry import LLM_ADAPTERS

        DeployCLI.cmd_list(args)
        # stdout = the listing itself, matching hale-deploy list
        print(f"{LLM_ADAPTERS.name}s (scope {LLM_ADAPTERS.scope}):")
        for key in LLM_ADAPTERS:
            print(f"  {LLM_ADAPTERS.scope}.{key}")
        return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point of ``hale-llm-deploy``; returns the process exit code."""
    return HaleLLMDeployCLI().main(argv)
