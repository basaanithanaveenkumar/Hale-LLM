"""Errors raised by the Jarvislabs integration."""


class CloudError(RuntimeError):
    """A cloud step failed (missing SDK, remote command failed, bad settings)."""
