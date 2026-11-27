"""Import every ``models_*`` subpackage so its decorators register the variants.

Adding a model variant: a new folder ``models/models_<name>/`` (model, loss, sampler and
collate, each registered under the variant name), one import here, one YAML under configs/.
"""

from hale_llm.models import models_autoregressive as _autoregressive  # noqa: F401
from hale_llm.models import models_mtp as _mtp  # noqa: F401
