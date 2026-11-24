"""Error raised when TorchTitan cannot run on this machine."""


class TorchTitanUnavailableError(RuntimeError):
    """TorchTitan needs NVIDIA GPUs and an installed ``torchtitan``; this machine has neither."""
