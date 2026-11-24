from hale_core.nn.optim import *  # noqa: F403
from hale_core.nn.optim import __all__ as _core_all

# Overrides hale_core's build_optimizer: takes the model (not its parameters) so
# optimizers such as Muon can split parameters by role.
from hale_llm.optim import MuonOptimizer, build_optimizer

__all__ = [*_core_all, "MuonOptimizer", "build_optimizer"]
