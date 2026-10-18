"""Autoregressive variant: causal next-token prediction (registers model, loss, sampler)."""

from hale_llm.models.models_autoregressive import loss as _loss  # noqa: F401
from hale_llm.models.models_autoregressive import model as _model  # noqa: F401
from hale_llm.models.models_autoregressive import sample as _sample  # noqa: F401

__all__: list[str] = []  # importing this package registers the variant
