"""Looped transformers: a stable (Parcae) injection, per-sequence depth sampling, and the looped backbone."""

from hale_llm.nn.looped.looped_backbone import LoopedBackbone
from hale_llm.nn.looped.recurrence_sampler import RecurrenceSampler
from hale_llm.nn.looped.stable_injection import StableInjection

__all__ = ["LoopedBackbone", "RecurrenceSampler", "StableInjection"]
