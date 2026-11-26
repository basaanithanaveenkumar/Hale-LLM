"""The lean trainer: one readable file in the nanochat / modded-nanogpt style.

This is the default backend (``train.trainer: lean``). It does only what a training run
on one device needs, so you can follow a whole run from top to bottom:

    1. read the config and pick the device
    2. build the data loaders
    3. build the model (optionally ``torch.compile`` it)
    4. build the optimizer from the registry (adamw or muon)
    5. define the learning-rate schedule: linear warmup, flat, linear warmdown
    6. loop: accumulate gradients over micro-batches, clip, update, log
    7. every N steps (and at the start and end): measure validation loss; log everything to TensorBoard;
       keep the best N checkpoints by validation score (``keep_best``, ``best_by``)
    8. every ``save_every`` steps write a rolling resume checkpoint; ``train.resume`` continues from it
    9. save a checkpoint that ``hale-llm-chat`` and ``hale-llm-sample`` can load

Every registered variant works because the loss comes from the loss registry.

Settings live under ``train.lean`` in the YAML (accum_steps, warmup_steps, warmdown_frac,
eval_every, eval_batches, save_every, keep_best, best_by, precision, compile).
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from collections.abc import Callable

import torch
from loguru import logger

import hale_llm.models  # noqa: F401  (registers the model variants)
from hale_llm.config.schema import RunConfig
from hale_llm.runtime import CheckpointStore
from hale_llm.registry import get_loss, get_variant
from hale_llm.runtime import move_batch_to_device
from hale_llm.dataloader.data_module import DataModule
from hale_llm.nn.moe import auxiliary_loss
from hale_llm.optim import build_optimizer
from hale_llm.utils.device import get_device
from hale_llm.training.duration import format_duration
from hale_llm.training.run_timer import RunTimer
from hale_llm.training.context_schedule import ContextSchedule
from hale_llm.training.context_stage_plan import ContextStagePlan
from hale_llm.tokenizer.store import TokenizerStore
from hale_llm.training.schedule import TrainSchedule
from hale_llm.training.mixed_precision import MixedPrecision
from hale_llm.training.resumable_batches import ResumableBatches
from hale_llm.training.resume_checkpoint import ResumeCheckpoint
from hale_llm.training.top_k_checkpoints import TopKCheckpoints
from hale_llm.training.training_logger import TrainingLogger
from hale_llm.metrics.base import MetricContext
from hale_llm.metrics.llm.accuracy import TokenAccuracyMetric


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


@torch.no_grad()
def validation_accuracy(model, loss_fn, loader, device: str, max_batches: int) -> float:
    """Greedy next-token accuracy over at most ``max_batches`` validation batches."""
    model.eval()
    metric = TokenAccuracyMetric()
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        batch = move_batch_to_device(batch, device)
        metric.update(MetricContext(model=model, batch=batch, loss=loss_fn(model, batch), variant=""))
    model.train()
    return metric.compute()["token_accuracy"]


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


def _keep_tokenizer_copy(cfg: RunConfig) -> None:
    """Put a copy of the tokenizer in the run folder when the run uses a shared one.

    The checkpoint then stays usable (chat, sample, eval look for ``<run>/tokenizer`` first)
    even if the shared tokenizer is later replaced.
    """
    if not cfg.experiment.enabled or cfg.tokenizer.source != "trained":
        return
    run_copy = Path(cfg.train.checkpoint_path).resolve().parent.parent / "tokenizer"
    if Path(cfg.tokenizer.path).resolve() != run_copy and not TokenizerStore.exists(run_copy):
        TokenizerStore.copy(cfg.tokenizer.path, run_copy)


def train_lean(cfg: RunConfig, tokenizer=None, on_step: Callable[[int], None] | None = None) -> list[float]:
    """Run the whole training loop and return the per-step training losses.

    Loop settings come from ``cfg.train.lean`` (see :class:`LeanConfig`). ``on_step(step)``,
    if given, is called after every optimizer step (the dry run uses it for timing).
    """
    # 1. Config, seed, device. One optimizer step uses ``accum`` micro-batches.
    lean = cfg.train.lean
    accum, warmup_steps, warmdown_frac = lean.accum_steps, lean.warmup_steps, lean.warmdown_frac
    eval_every, eval_batches, compile_model = lean.eval_every, lean.eval_batches, lean.compile
    keep_best, best_by, save_every = lean.keep_best, lean.best_by, lean.save_every
    torch.manual_seed(cfg.train.seed)
    device = get_device(cfg.device)
    precision = MixedPrecision.from_setting(lean.precision, device)  # fp32 | bf16 | fp16 (+ loss scaling)
    # 1b. The resume checkpoint lives next to the final one. A run that already finished is not redone.
    resume = ResumeCheckpoint(Path(cfg.train.checkpoint_path).parent)
    finished_at = resume.completed_step() if cfg.train.resume else None
    if finished_at is not None:
        logger.info("this run already finished at step {}; nothing to do (use --no-resume to train again)", finished_at)
        return []
    # 2. Data. ``endless`` turns the loader into an infinite stream of micro-batches.
    #    The step count comes from train.steps, or from train.epochs x batches per epoch.
    data = DataModule(cfg, tokenizer)
    train_loader = data.train_loader()
    # One optimizer step consumes ``accum`` batches, so an epoch is len(loader) // accum steps.
    total_steps = TrainSchedule.from_config(cfg, max(len(train_loader) // accum, 1)).total_steps
    # Optional context-extension stages (empty list = one stage for the whole run).
    schedule = ContextSchedule.from_config(cfg, total_steps)
    current_plan = schedule.plan_at(0)
    val_loader = data.val_loader()
    _keep_tokenizer_copy(cfg)

    logger.info(
        "lean trainer | variant={} device={} steps={} micro-batch={} accum={} "
        "(effective batch {}) precision={}",
        cfg.variant, device, total_steps, cfg.train.batch_size, accum,
        cfg.train.batch_size * accum, precision.name,
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
    best = TopKCheckpoints(
        Path(cfg.train.checkpoint_path).parent / "best", keep_best, best_by,
        higher_is_better=best_by != "loss",
    )

    def checkpoint_state(step: int, with_optimizer: bool = True) -> dict:
        """Everything chat / sample / eval need; the best-N files skip the optimizer to stay small."""
        state = dict(
            model_state_dict=plain_model.state_dict(), variant=cfg.variant,
            vocab_size=len(data.tokenizer), epoch=0, step=step, config=cfg.model_dump(),
        )
        if with_optimizer:
            state["optimizer_state_dict"] = optimizer.state_dict()
        return state

    # 4b. Resume an interrupted run (train.resume): weights, optimizer, loss scale, random state, step
    #     and data position come back from resume.pt. Without it (or when none exists) start at step 0.
    start_step, tokens_seen, start_batches, earlier_seconds = 0, 0, 0, 0.0
    state = resume.load() if cfg.train.resume else None
    resumed = state is not None
    if resumed:
        resume.apply(state, model=plain_model, optimizer=optimizer, precision=precision,
                     variant=cfg.variant, vocab_size=len(data.tokenizer))
        start_step, tokens_seen, start_batches = state["step"], state["tokens_seen"], state["stream_batches"]
        earlier_seconds = state.get("elapsed_seconds", 0.0)  # missing in checkpoints written before this was recorded
        best.restore_from_disk(max_step=start_step)
        logger.info("resumed from {} at step {}/{} ({:,} tokens already seen, {} of training so far)",
                    resume.path, start_step, total_steps, tokens_seen, format_duration(earlier_seconds))
        if state["total_steps"] != total_steps:
            logger.warning("the run was {} steps long when saved, now {}: the lr schedule changes", state["total_steps"], total_steps)
        del state  # the optimizer state is large; free the CPU copy
    else:
        if cfg.train.resume:
            logger.info("resume requested but {} does not exist: starting from step 0", resume.path)
        resume.clear()  # a fresh start must not leave an old resume.pt for a later restart to pick up

    # The data position was saved inside the stage that was running when the checkpoint was written.
    position_plan = schedule.plan_at(max(start_step - 1, 0))

    def batch_stream(loader, plan) -> ResumableBatches:
        """Reproducible batch stream for a stage; a resumed run continues inside its own stage."""
        return ResumableBatches(loader, cfg.train.seed, start_batches if plan is position_plan else 0)

    train_batches = batch_stream(train_loader, current_plan)

    losses: list[float] = []
    started = time.time()
    timer = RunTimer(earlier_seconds)  # total training time, also across interruptions
    balance_coef = cfg.model.moe_aux_loss_coef if getattr(plain_model, "uses_moe", False) else 0.0

    with TrainingLogger(cfg, purge_from_step=start_step + 1 if resumed else None) as tb:
        tb.log_run_start(cfg, plain_model, total_steps, n_windows)
        if eval_every and not resumed:  # the untrained model's validation loss is the starting point of the curve
            with precision.autocast():
                tb.log_validation(0, validation_loss(model, loss_fn, val_loader, device, eval_batches))

        for step in range(start_step, total_steps):
            # 5a. Entering a new context stage: longer window, larger RoPE base, fresh loaders.
            plan = schedule.plan_at(step)
            if plan is not current_plan:
                logger.info("step {}: entering context stage '{}' (window {})", step, plan.name, plan.context_window)
                apply_stage_to_model(plain_model, cfg, plan)
                data, train_loader, val_loader = stage_loaders(cfg, plan, data.tokenizer)
                train_batches = batch_stream(train_loader, plan)
                current_plan = plan
                logger.info(
                    "stage data: {:,} context windows of {} tokens",
                    len(train_loader.dataset), plan.context_window,
                )
            step_started = time.perf_counter()

            # 5b. A model whose loss depends on how far training has come (the MTP loss weight) is told.
            if hasattr(plain_model, "set_training_progress"):
                plain_model.set_training_progress(step / total_steps)

            # 5c. Learning rate for this step: every group keeps its own base lr, scaled.
            scale = lr_multiplier(step, total_steps, warmup_steps, warmdown_frac)
            for group in optimizer.param_groups:
                group["lr"] = group["base_lr"] * scale

            # 6. Accumulate gradients. Each micro-batch loss is divided by ``accum`` so the
            #    summed gradient equals the gradient of the mean over the big batch.
            optimizer.zero_grad(set_to_none=True)
            step_loss = 0.0
            balance_loss = 0.0  # MoE load-balancing term, averaged over the micro-batches
            part_sums: dict[str, torch.Tensor] = {}  # pieces of the loss a model reports (e.g. MTP), averaged the same way
            for _ in range(accum):
                batch = move_batch_to_device(next(train_batches), device)
                with precision.autocast():
                    loss = loss_fn(model, batch) / accum
                precision.backward(loss)  # fp16: the loss is scaled up first
                step_loss += loss.item()
                for name, value in getattr(plain_model, "loss_parts", {}).items():
                    part_sums[name] = part_sums.get(name, 0.0) + value / accum
                if balance_coef:
                    balance_loss += float(auxiliary_loss(plain_model.backbone).detach()) / accum
            if not math.isfinite(step_loss):
                raise FloatingPointError(f"non-finite loss at step {step}: {step_loss}")

            # 7. Clip the global gradient norm, then update every parameter.
            #    fp16 also unscales the gradients first and skips the update on overflow.
            grad_norm = precision.clip_and_step(optimizer, plain_model.parameters(), cfg.train.grad_clip)
            losses.append(step_loss)
            if on_step is not None:
                on_step(step)

            # 7b. Record the step. ``ce_loss`` is the next-token loss alone: the optimised loss
            #     also contains the MoE balancing term, which validation does not.
            step_seconds = time.perf_counter() - step_started
            step_tokens = current_plan.batch_size * accum * current_plan.context_window
            tokens_seen += step_tokens
            ce_loss = step_loss - balance_coef * balance_loss
            if "ntp_loss" in part_sums:  # the model reports its next-token loss directly (MTP variant)
                ce_loss = float(part_sums["ntp_loss"])
            scalars = {
                "train/loss": step_loss,
                "train/ce_loss": ce_loss,
                "train/perplexity": math.exp(min(ce_loss, 20.0)),
                "train/bits_per_token": ce_loss / math.log(2),
                "train/grad_norm": grad_norm,
                "train/grad_clipped": float(grad_norm > cfg.train.grad_clip),
                "train/lr_scale": scale,
                "train/tokens_seen": tokens_seen,
                "train/tokens_per_sec": step_tokens / step_seconds,
                "train/step_seconds": step_seconds,
                "train/epoch": (step + 1) * accum / len(train_loader),
                "context/window": current_plan.context_window,
            }
            for name, value in part_sums.items():
                scalars[f"train/{name}"] = float(value)
            for index, group in enumerate(optimizer.param_groups):
                scalars["train/lr" if index == 0 else f"train/lr_group{index}"] = group["lr"]
            if precision.scaler.is_enabled():
                scalars["train/loss_scale"] = precision.loss_scale
                scalars["train/step_skipped"] = float(precision.step_skipped)
            if balance_coef:
                # The optimised term sums the layers; log the per-layer mean, where 1.0 = even routing.
                scalars["train/moe_balance_loss"] = balance_loss / cfg.model.n_layers
            tb.log_step(step + 1, scalars)

            last_step = step == total_steps - 1
            if step % cfg.train.log_every == 0 or last_step:
                tb.log_slow_stats(step + 1, plain_model, device)
                logger.info(
                    "step {}/{} loss={:.4f} lr_scale={:.3f} grad_norm={:.3f} {:.0f} tok/s {:.1f}s",
                    step + 1, total_steps, step_loss, scale, grad_norm,
                    scalars["train/tokens_per_sec"], time.time() - started,
                )
            if eval_every and ((step + 1) % eval_every == 0 or last_step):
                with precision.autocast():
                    val_loss = validation_loss(model, loss_fn, val_loader, device, eval_batches)
                tb.log_validation(step + 1, val_loss)
                logger.info("step {} validation loss = {:.4f} (perplexity {:.1f})", step + 1, val_loss, math.exp(min(val_loss, 20.0)))
                # Rank this checkpoint by the chosen validation score; keep only the best N.
                score = val_loss
                if best_by == "token_accuracy":
                    with precision.autocast():
                        score = validation_accuracy(model, loss_fn, val_loader, device, eval_batches)
                    tb.log_step(step + 1, {"val/token_accuracy": score})
                    logger.info("step {} validation token accuracy = {:.4f}", step + 1, score)
                state = checkpoint_state(step + 1, with_optimizer=False)
                state.pop("step")  # offer() records the step itself
                best.offer(score, step + 1, **state)

            # 8. Rolling resume checkpoint, so an interruption (a spot instance being paused)
            #    costs at most ``save_every`` steps instead of the whole run.
            if save_every and (step + 1) % save_every == 0 and not last_step:
                resume.save(step=step + 1, tokens_seen=tokens_seen, stream_batches=train_batches.consumed,
                            total_steps=total_steps, variant=cfg.variant, vocab_size=len(data.tokenizer),
                            model=plain_model, optimizer=optimizer, precision=precision,
                            elapsed_seconds=timer.total_seconds())

    # 9. Save a checkpoint that chat / sample / eval can load, then record that the run is finished.
    store.save(cfg.train.checkpoint_path, **checkpoint_state(total_steps))
    total_seconds = timer.total_seconds()
    resume.mark_complete(total_steps, total_seconds)
    logger.info("saved checkpoint to {}", cfg.train.checkpoint_path)
    # 10. How long the whole run took. After an interruption this counts the earlier sessions too.
    logger.info(
        "training finished: {} steps, {:,} tokens in {} ({:.1f} s){}",
        total_steps, tokens_seen, format_duration(total_seconds), total_seconds,
        f"; this session {format_duration(timer.session_seconds())}" if earlier_seconds else "",
    )
    return losses
