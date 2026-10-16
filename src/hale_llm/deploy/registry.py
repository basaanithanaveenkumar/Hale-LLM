"""``hale_llm`` scope of the deployment adapter registry (mmdet-style child registry).

Adapters registered here are addressed as ``hale_llm.<ClassName>`` from configs, e.g.::

    adapter:
      type: hale_llm.AutoregressiveLMAdapter
      config: configs/autoregressive/nano.yaml

The child falls back to the root ``ADAPTERS`` registry of HaleBlocks, so generic adapters
(``TorchModuleAdapter`` ...) stay reachable through it.
"""

from __future__ import annotations

from hale_core.deploy.registry import ADAPTERS, Registry

LLM_ADAPTERS = Registry("adapter", parent=ADAPTERS, scope="hale_llm")

__all__ = ["LLM_ADAPTERS"]
