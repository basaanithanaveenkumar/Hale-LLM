"""Load a checkpointed variant and run the registered sampler."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from loguru import logger
from torch import nn

from hale_llm.config.run import VARIANT_FFN_TYPE
from hale_llm.config.schema import RunConfig
from hale_llm.runtime import CheckpointStore
from hale_llm.registry import get_sampler, get_variant
from hale_llm.runtime import log_model_summary
from hale_llm.dataloader.tokenizer import build_tokenizer_for
from hale_llm.inference.encode import encode_prompt
from hale_llm.inference.stop import cut_at_stop
from hale_llm.tokenizer.store import TokenizerStore
from hale_llm.utils.device import get_device


@dataclass
class GenerationResult:
    token_ids: torch.Tensor
    text: str
    prompt: str
    history: list | None = None


@dataclass
class InferenceSession:
    model: nn.Module
    tokenizer: Any
    cfg: RunConfig
    device: str

    def generate(
        self,
        prompt: str | None = None,
        *,
        return_history: bool = False,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
    ) -> GenerationResult:
        prompt = prompt if prompt is not None else self.cfg.sample.prompt
        max_new_tokens = max_new_tokens if max_new_tokens is not None else self.cfg.sample.max_new_tokens
        temperature = temperature if temperature is not None else self.cfg.sample.temperature
        sampler = get_sampler(self.cfg.variant)
        self.model.eval()
        ids = encode_prompt(self.tokenizer, prompt, self.model, self.device)
        logger.info("inference variant={} prompt={!r} return_history={}", self.cfg.variant, prompt, return_history)
        out = sampler(
            self.model,
            self.tokenizer,
            ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            return_history=return_history,
        )
        history = None
        if return_history:
            token_ids, history = out
        else:
            token_ids = out
        # Stop at the end-of-text token.
        kept = cut_at_stop(token_ids[0], ids.size(1), getattr(self.tokenizer, "eos_token_id", None))
        text = self.tokenizer.decode(kept.tolist(), skip_special_tokens=True)
        logger.info("generated: {}", text)
        return GenerationResult(token_ids=token_ids, text=text, prompt=prompt, history=history)


def session_from_model(model: nn.Module, tokenizer, cfg: RunConfig, device: str | None = None) -> InferenceSession:
    device = device or get_device(cfg.device)
    return InferenceSession(model=model.to(device), tokenizer=tokenizer, cfg=cfg, device=device)


#: Model settings that are inference-time choices, so the YAML keeps control of them.
_INFERENCE_KNOBS = ("rope_scaling", "rope_scaling_factor", "rope_original_length")


def _adopt_trained_model(cfg: RunConfig, ckpt: dict) -> None:
    """Rebuild the model exactly as it was trained, whatever the YAML passed to the CLI says.

    A checkpoint only fits the architecture it was trained with (width, depth, experts, norm,
    RoPE base, context window). gpt-oss does the same: it reads ``config.json`` from the
    checkpoint folder. Only the RoPE-scaling fields stay under the YAML's control, because
    running past the trained length with YaRN is something you choose at inference time.
    """
    trained = (ckpt.get("config") or {}).get("model")
    if not trained:
        return  # an old checkpoint without a saved config: trust the YAML
    if ckpt.get("variant") not in (None, cfg.variant):
        raise ValueError(f"checkpoint was trained as variant {ckpt['variant']!r}, config says {cfg.variant!r}")
    kept = {name: getattr(cfg.model, name) for name in _INFERENCE_KNOBS}
    settings = {**trained, **kept}
    # Checkpoints saved before RunConfig fixed the feed-forward type per variant may say "mlp".
    settings["ffn_type"] = VARIANT_FFN_TYPE.get(cfg.variant, settings["ffn_type"])
    adopted = type(cfg.model)(**settings)
    changed = sorted(n for n in type(cfg.model).model_fields if getattr(adopted, n) != getattr(cfg.model, n))
    if changed:
        logger.info("using the trained model settings from the checkpoint; differs from the YAML in: {}", changed)
    cfg.model = adopted
    cfg.data.context_window = adopted.max_length


def load_session(
    cfg: RunConfig,
    checkpoint_path: str | None = None,
    *,
    require_checkpoint: bool = False,
    tokenizer=None,
    checkpoints: CheckpointStore | None = None,
) -> InferenceSession:
    """Load weights if the checkpoint exists.

    Sample/viz default to an untrained model when the configured path is missing.
    Pass ``require_checkpoint=True`` (eval) or an explicit ``checkpoint_path`` that
    does not exist to keep the old FileNotFoundError.
    """
    import hale_llm.models  # noqa: F401

    device = get_device(cfg.device)
    path = checkpoint_path or cfg.train.checkpoint_path
    # A checkpoint lives in <run>/checkpoints/; its tokenizer in <run>/tokenizer/.
    beside_checkpoint = Path(path).resolve().parent.parent / "tokenizer"
    if checkpoint_path and TokenizerStore.exists(beside_checkpoint):
        cfg.tokenizer.path = str(beside_checkpoint)
    # Inference must never train a tokenizer: it has to be the one the model was trained with.
    tokenizer = tokenizer or build_tokenizer_for(cfg, allow_training=False)
    store = checkpoints or CheckpointStore()
    model_cls = get_variant(cfg.variant)

    if store.exists(path):
        ckpt = store.load(path, device)
        _adopt_trained_model(cfg, ckpt)
        model = model_cls(vocab_size=ckpt["vocab_size"], cfg=cfg).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        logger.info("inference session variant={} device={} ckpt={}", cfg.variant, device, path)
        log_model_summary(model, title=f"model summary  variant={cfg.variant}")
        return InferenceSession(model=model, tokenizer=tokenizer, cfg=cfg, device=device)

    explicit_missing = checkpoint_path is not None
    if require_checkpoint or explicit_missing:
        store.load(path, device)

    logger.warning(
        "no checkpoint at {}; using untrained {}. "
        "Train first, or pass --checkpoint PATH",
        path,
        cfg.variant,
    )
    model = model_cls(vocab_size=len(tokenizer), cfg=cfg).to(device)
    log_model_summary(model, title=f"model summary  variant={cfg.variant}")
    return InferenceSession(model=model, tokenizer=tokenizer, cfg=cfg, device=device)
