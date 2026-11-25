"""The lean trainer: one readable file in the nanochat / modded-nanogpt style.

This is the default backend (``train.trainer: lean``). It does only what a training run
on one device needs, so you can follow a whole run from top to bottom:

    1. read the config and pick the device
    2. build the data loaders
    3. build the model (optionally ``torch.compile`` it)
    4. build the optimizer from the registry (adamw or muon)
    5. define the learning-rate schedule: linear warmup, flat, linear warmdown
    6. loop: accumulate gradients over micro-batches, clip, update, log
    7. every N steps: measure validation loss
    8. save a checkpoint that ``hale-llm-chat`` and ``hale-llm-sample`` can load

Every registered variant works because the loss comes from the loss registry.

Settings live under ``train.lean`` in the YAML (accum_steps, warmup_steps, warmdown_frac,
eval_every, eval_batches, compile). It has no resume support.
"""

from __future__ import annotations

import contextlib
import math
import time
from collections.abc import Iterator

import torch
from loguru import logger

import hale_llm.models  # noqa: F401  (registers the model variants)
from hale_llm.config.schema import RunConfig
from hale_llm.runtime import CheckpointStore
from hale_llm.registry import get_loss, get_variant
from hale_llm.runtime import move_batch_to_device
from hale_llm.dataloader.data_module import DataModule
from hale_llm.optim import build_optimizer
from hale_llm.utils.device import get_device
from hale_llm.training.context_schedule import ContextSchedule
from hale_llm.training.context_stage_plan import ContextStagePlan
from hale_llm.training.schedule import TrainSchedule


def lr_multiplier(step: int, total_steps: int, warmup_steps: int, warmdown_frac: float) -> float:
    """Factor applied to the base learning rate at ``step`` (0-based).

    Shape: ramps 0 -> 1 over ``warmup_steps``, stays at 1, then ramps 1 -> 0 over the
    last ``warmdown_frac`` of training. Tiny example with total=100, warmup=10,
    warmdown_frac=0.2: step 5 -> 0.55, step 50 -> 1.0, step 90 -> 0.5, step 99 -> 0.05.
    """
    if step < warmup_steps:
        return (step + 1) / warmup_steps
    warmdown_start = total_steps * (1.0 - warmdown_frac)
    if step < warmdown_start:
        return 1.0
    return max(0.0, (total_steps - step) / max(total_steps - warmdown_start, 1))


def endless(loader) -> Iterator[dict]:
    """Yield batches forever, restarting the loader when an epoch ends."""
    while True:
        yield from loader


@torch.no_grad()
def validation_loss(model, loss_fn, loader, device: str, max_batches: int) -> float:
    """Mean loss over at most ``max_batches`` validation batches."""
    model.eval()
    losses = []
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        losses.append(loss_fn(model, move_batch_to_device(batch, device)).item())
    model.train()
    return sum(losses) / max(len(losses), 1)


def apply_stage_to_model(model, cfg: RunConfig, plan: ContextStagePlan) -> None:
    """Switch the model and config to a stage's context window and RoPE base.

    RoPE has no position table, so a longer window only needs the backbone's length limit
    raised and the rotation base changed. ``cfg`` is updated too because the checkpoint
    stores it, and inference must rebuild the model with the *final* stage's settings.
    """
    cfg.set_context_window(plan.context_window)
    cfg.model.rope_theta = plan.rope_theta
    model.backbone.max_length = plan.context_window
    model.backbone.positional.set_theta(plan.rope_theta)


def stage_loaders(cfg: RunConfig, plan: ContextStagePlan, tokenizer):
    """Train and validation loaders for one stage (its own window, batch and long-document mix)."""
    stage_cfg = cfg.model_copy(deep=True)
    stage_cfg.set_context_window(plan.context_window)
    stage_cfg.train.batch_size = plan.batch_size
    stage_cfg.data.long_doc_fraction = plan.long_doc_fraction
    stage_cfg.data.long_doc_min_tokens = plan.long_doc_min_tokens
    data = DataModule(stage_cfg, tokenizer)
    return data, data.train_loader(), data.val_loader()


def train_lean(cfg: RunConfig, tokenizer=None) -> list[float]:
    """Run the whole training loop and return the per-step training losses.

    Loop settings come from ``cfg.train.lean`` (see :class:`LeanConfig`).
    """
    # 1. Config, seed, device. One optimizer step uses ``accum`` micro-batches.
    lean = cfg.train.lean
    accum, warmup_steps, warmdown_frac = lean.accum_steps, lean.warmup_steps, lean.warmdown_frac
    eval_every, eval_batches, compile_model = lean.eval_every, lean.eval_batches, lean.compile
    torch.manual_seed(cfg.train.seed)
    device = get_device(cfg.device)
    use_bf16 = device == "cuda"  # bf16 autocast is reliable on CUDA; MPS/CPU stay fp32
    # 2. Data. ``endless`` turns the loader into an infinite stream of micro-batches.
    #    The step count comes from train.steps, or from train.epochs x batches per epoch.
    data = DataModule(cfg, tokenizer)
    train_loader = data.train_loader()
    total_steps = TrainSchedule.from_config(cfg, len(train_loader)).total_steps
    train_batches = endless(train_loader)
    # Optional context-extension stages (empty list = one stage for the whole run).
    schedule = ContextSchedule.from_config(cfg, total_steps)
    current_plan = schedule.plan_at(0)
    val_loader = data.val_loader()

    logger.info(
        "lean trainer | variant={} device={} steps={} micro-batch={} accum={} "
        "(effective batch {}) bf16={}",
        cfg.variant, device, total_steps, cfg.train.batch_size, accum,
        cfg.train.batch_size * accum, use_bf16,
    )
    n_windows = len(train_loader.dataset)
    logger.info(
        "training data: {:,} context windows of {} tokens = {:,} tokens | {} batches per epoch",
        n_windows, cfg.model.max_length, n_windows * cfg.model.max_length, len(train_loader),
    )

    # 3. Model. Keep a handle on the plain module: a compiled wrapper renames the
    #    state-dict keys, and checkpoints must stay loadable by hale-llm-chat.
    model = get_variant(cfg.variant)(vocab_size=len(data.tokenizer), cfg=cfg).to(device)
    loss_fn = get_loss(cfg.variant)
    plain_model = model
    if compile_model:
        logger.info("compiling the model with torch.compile (first steps will be slow)")
        model = torch.compile(model)
    model.train()

    # 4. Optimizer from the registry, plus each group's base lr so the schedule can scale it.
    optimizer = build_optimizer(
        cfg.train.optimizer_type,
        plain_model,
        {
            "lr": cfg.train.lr,
            "weight_decay": cfg.train.weight_decay,
            "adam_beta1": cfg.train.adam_beta1,
            "adam_beta2": cfg.train.adam_beta2,
            "muon_lr": cfg.train.muon_lr,
            "muon_momentum": cfg.train.muon_momentum,
        },
    )
    for group in optimizer.param_groups:
        group["base_lr"] = group["lr"]

    store = CheckpointStore()
    losses: list[float] = []
    started = time.time()

    for step in range(total_steps):
        # 5a. Entering a new context stage: longer window, larger RoPE base, fresh loaders.
        plan = schedule.plan_at(step)
        if plan is not current_plan:
            logger.info("step {}: entering context stage '{}' (window {})", step, plan.name, plan.context_window)
            apply_stage_to_model(plain_model, cfg, plan)
            data, train_loader, val_loader = stage_loaders(cfg, plan, data.tokenizer)
            train_batches = endless(train_loader)
            current_plan = plan
            logger.info(
                "stage data: {:,} context windows of {} tokens",
                len(train_loader.dataset), plan.context_window,
            )

        # 5b. Learning rate for this step: every group keeps its own base lr, scaled.
        scale = lr_multiplier(step, total_steps, warmup_steps, warmdown_frac)
        for group in optimizer.param_groups:
            group["lr"] = group["base_lr"] * scale

        # 6. Accumulate gradients. Each micro-batch loss is divided by ``accum`` so the
        #    summed gradient equals the gradient of the mean over the big batch.
        optimizer.zero_grad(set_to_none=True)
        step_loss = 0.0
        for _ in range(accum):
            batch = move_batch_to_device(next(train_batches), device)
            autocast = (
                torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                if use_bf16
                else contextlib.nullcontext()
            )
            with autocast:
                loss = loss_fn(model, batch) / accum
            loss.backward()
            step_loss += loss.item()
        if not math.isfinite(step_loss):
            raise FloatingPointError(f"non-finite loss at step {step}: {step_loss}")

        # 7. Clip the global gradient norm, then update every parameter.
        grad_norm = torch.nn.utils.clip_grad_norm_(plain_model.parameters(), cfg.train.grad_clip)
        optimizer.step()
        losses.append(step_loss)

        if step % cfg.train.log_every == 0 or step == total_steps - 1:
            elapsed = time.time() - started
            logger.info(
                "step {}/{} loss={:.4f} lr_scale={:.3f} grad_norm={:.3f} {:.1f}s",
                step + 1, total_steps, step_loss, scale, float(grad_norm), elapsed,
            )
        if eval_every and (step + 1) % eval_every == 0:
            logger.info(
                "step {} validation loss = {:.4f}",
                step + 1, validation_loss(model, loss_fn, val_loader, device, eval_batches),
            )

    # 8. Save a checkpoint that chat / sample / eval can load.
    store.save(
        cfg.train.checkpoint_path,
        model_state_dict=plain_model.state_dict(),
        optimizer_state_dict=optimizer.state_dict(),
        variant=cfg.variant,
        vocab_size=len(data.tokenizer),
        epoch=0,
        step=total_steps,
        config=cfg.model_dump(),
    )
    logger.info("saved checkpoint to {}", cfg.train.checkpoint_path)
    return losses
