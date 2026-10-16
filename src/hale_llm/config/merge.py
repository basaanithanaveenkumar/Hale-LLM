"""Nested dict merge for YAML inheritance (re-export of HaleBlocks' implementation).

:func:`deep_merge` drops the inheritance keys (``inherits``, ``_base_``) of the override and honours
``_delete_: true`` (replace instead of merge); see :class:`hale_core.config.dict_merger.DictMerger`.
"""

from hale_core.config.merge import deep_merge

__all__ = ["deep_merge"]
