"""Base adapter of every HaloDiffusionLLM variant (Template Method over ``ModelAdapter``)."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from functools import cached_property
from typing import Any, ClassVar

import torch
from hale_core.deploy.adapters.model_adapter import ModelAdapter
from hale_core.deploy.core.io_signature import IOSignature
from hale_core.deploy.exceptions import ConfigError
from loguru import logger
from torch import nn

from hale_llm.deploy.adapters.calibration_corpus import CalibrationCorpus
from hale_llm.deploy.adapters.run_config_resolver import RunConfigResolver


class HaleLLMAdapter(ModelAdapter):
    """Shared plumbing: run config → registered variant → named-tensor export contract.

    Subclasses set :attr:`variant` and implement :meth:`signature_inputs`,
    :meth:`forward_for_export` and (optionally) :meth:`calibration_inputs`.

    Vocabulary size resolution (first hit wins): the ``vocab_size`` argument, the
    ``vocab_size`` stored in the trainer checkpoint, the length of the configured tokenizer
    (with its added ``[MASK]`` token), exactly as the trainer computes it.

    Args:
        config: run-config YAML the model was trained with.
        checkpoint: trainer checkpoint (``model_state_dict`` inside) or a bare state dict.
        vocab_size: explicit vocabulary size.
        overrides: deep-merged onto the run config (tiny CI models, loop counts ...).
        max_batch: largest batch the exported graph must accept.
        opt_seq: sequence length used for tracing / engine tuning (capped at ``max_length``).
        dynamic_seq: export a dynamic sequence dimension (``False`` pins it to ``max_length``,
            which is what the fixed-canvas diffusion samplers use).
    """

    variant: ClassVar[str]
    outputs: ClassVar[tuple[str, ...]] = ("logits",)

    def __init__(
        self,
        config: str,
        checkpoint: str | None = None,
        vocab_size: int | None = None,
        overrides: Mapping[str, Any] | None = None,
        max_batch: int = 8,
        opt_seq: int = 64,
        dynamic_seq: bool = True,
        **base: Any,
    ) -> None:
        super().__init__(checkpoint=checkpoint, **base)
        self.resolver = RunConfigResolver(config, overrides)
        self._vocab_size = vocab_size
        self.max_batch = max_batch
        self.opt_seq = opt_seq
        self.dynamic_seq = dynamic_seq

    # ------------------------------------------------------------------ config / model
    @cached_property
    def run_config(self):
        """The resolved, validated run config (overrides applied).

        Raises:
            hale_core.exceptions.ConfigError: if the YAML's ``variant`` is not this adapter's.
        """
        cfg = self.resolver.load()
        if cfg.variant != self.variant:
            raise ConfigError(
                f"{type(self).__name__} deploys variant {self.variant!r}, but "
                f"{self.resolver.path.name} is {cfg.variant!r}"
            )
        return cfg

    @cached_property
    def vocab_size(self) -> int:
        """Vocabulary size: explicit argument, then checkpoint metadata, then the offline tokenizer.

        Raises:
            hale_core.exceptions.ConfigError: if none of the three is available.
        """
        if self._vocab_size is not None:
            return int(self._vocab_size)
        if self.checkpoint:
            meta = torch.load(self.checkpoint, map_location="cpu", weights_only=False)
            if isinstance(meta, Mapping) and "vocab_size" in meta:
                return int(meta["vocab_size"])
        tokenizer = CalibrationCorpus(self.run_config.data.tokenizer_name).tokenizer()
        if tokenizer is None:
            raise ConfigError(
                "cannot infer vocab_size: pass adapter.vocab_size, a checkpoint with "
                "'vocab_size', or make the tokenizer available offline"
            )
        return len(tokenizer)

    @property
    def max_length(self) -> int:
        """``model.max_length`` of the run config (the largest exportable sequence)."""
        return int(self.run_config.model.max_length)

    def build_model(self) -> nn.Module:
        """Instantiate the registered variant with the resolved config and vocabulary."""
        import hale_llm.models  # noqa: F401 - registers the variants
        from hale_llm.core.registry import get_variant

        logger.info(
            "building {} (vocab={}, d_model={}, layers={})",
            self.variant,
            self.vocab_size,
            self.run_config.model.d_model,
            self.run_config.model.n_layers,
        )
        return get_variant(self.variant)(vocab_size=self.vocab_size, cfg=self.run_config)

    # ------------------------------------------------------------------ signature
    def signature_inputs(self) -> list[dict[str, Any]]:
        """Input tensor specs in positional order (see :class:`TensorSpec` config keys)."""
        raise NotImplementedError

    def token_spec(self, name: str) -> dict[str, Any]:
        """Spec of an int64 ``[batch, seq]`` token input called ``name`` with values in ``[0,
        vocab)``.
        """
        return {
            "name": name,
            "shape": ["batch", "seq"],
            "dtype": "int64",
            "value_range": [0, self.vocab_size],
        }

    def default_signature(self) -> IOSignature:
        """Export signature: dynamic batch ``1…max_batch``; sequence dynamic or pinned to
        ``max_length``.
        """
        seq_max = self.max_length
        seq_opt = min(self.opt_seq, seq_max)
        seq = [1, seq_opt, seq_max] if self.dynamic_seq else seq_max
        return IOSignature.from_config(
            {
                "inputs": self.signature_inputs(),
                "outputs": list(self.outputs),
                "dims": {"batch": [1, 1, self.max_batch], "seq": seq},
            }
        )

    # ------------------------------------------------------------------ calibration
    def calibration_inputs(self, token_ids: torch.Tensor) -> dict[str, torch.Tensor]:
        """Map a batch of real token ids to this variant's inputs (default: ``input_ids``)."""
        return {"input_ids": token_ids}

    def calibration_batches(self, num_batches: int) -> Iterator[dict[str, torch.Tensor]] | None:
        """``num_batches`` real-text calibration inputs, or ``None`` when the tokenizer is not
        cached.

        The pipeline then falls back to random calibration data.
        """
        sig = self.signature()
        seq = sig.dims["seq"].opt
        corpus = CalibrationCorpus(self.run_config.data.tokenizer_name)
        windows = corpus.token_windows(seq, num_batches, self.vocab_size)
        if windows is None:
            return None
        return (self.calibration_inputs(ids) for ids in windows)
