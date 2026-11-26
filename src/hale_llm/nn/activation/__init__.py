"""Activation functions of the feed-forward layers. Importing this package registers the built-in names."""

from hale_llm.nn.activation.activation_spec import ActivationSpec
from hale_llm.nn.activation.registry import ACTIVATIONS, get_activation, register_activation

__all__ = ["ACTIVATIONS", "ActivationSpec", "get_activation", "register_activation"]
