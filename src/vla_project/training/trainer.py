"""Minimal Trainer with Accelerator-driven backward / loss gather.

Single-GPU `python scripts/train.py ...` and multi-GPU `accelerate launch
... scripts/train.py ...` use the same code path. Accelerator()'s no-arg
constructor reads env vars set by `accelerate launch`; in single-process
mode it is a near-no-op.
"""
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn

from vla_project.training.checkpoint import save_checkpoint


@dataclass
class TrainerConfig:
    max_steps: int = 100
    log_every: int = 10
    grad_clip_norm: float = 1.0
    gradient_accumulation_steps: int = 1
    save_every: Optional[int] = None  # save every N steps; None disables periodic
    save_dir: Optional[str] = None    # parent dir for step_<N>/ checkpoints
    # LR schedule: linear warmup over `warmup_steps`, then cosine decay to
    # `min_lr_ratio * init_lr`. Defaults disable scheduling (warmup=0,
    # min_lr_ratio=1.0 → constant LR per group).
    warmup_steps: int = 0
    min_lr_ratio: float = 1.0
    # X-VLA-style stage curriculum:
    #   - groups in `schedule_group_names`: get freeze (if also in
    #     freeze_group_names) → warmup → cosine decay (set min_lr_ratio=1.0
    #     to drop decay and keep flat-after-warmup).
    #   - all other groups: constant lr at their initial coef × base.
    # Defaults: backbone + soft prompts go through schedule; head /
    # projections / action queries stay at full lr the whole time.
    freeze_steps: int = 0
    freeze_group_names: Tuple[str, ...] = ("gemma_lora", "siglip")
    schedule_group_names: Tuple[str, ...] = ("gemma_lora", "siglip", "soft_prompts")
    # Keys whose float tensors stay in their original dtype (do NOT cast to
    # model_dtype). Default protects regression labels: bf16 target makes
    # L1/MSE loss subtly lossy. Add ``loss_weight``, ``return_to_go``, etc.
    # as appropriate for new tasks.
    keep_dtype_keys: Tuple[str, ...] = ("target_action",)
    # Diagnostic logging for the first N batches (B10): per-batch domain_id
    # histogram, wrist_mask presence rate, action / proprio min-max-mean. Helps
    # catch DA-row off-by-one, wrist_mask polarity bugs, normalization breakage
    # before they pollute many gradient steps. 0 disables (default — keep
    # legacy v33-v36 behavior). v37 OXE multi-domain sets 100.
    diagnostic_first_n_batches: int = 0


def _cast_tensor_to_device(t: torch.Tensor, device, model_dtype, keep_orig: bool) -> torch.Tensor:
    t = t.to(device)
    if t.is_floating_point() and not keep_orig:
        t = t.to(model_dtype)
    return t


def _cast_batch(
    batch: Any, device, model_dtype, keep_dtype_keys: Tuple[str, ...]
) -> Any:
    """Recursively move tensors in a batch to ``device`` and cast floats to
    ``model_dtype``. Keys listed in ``keep_dtype_keys`` retain their original
    float dtype (e.g. regression targets that the loss should compute in fp32).

    Handles nested dicts (e.g. ``{"obs": {"image": ..., "proprio": ...}, "action": ...}``)
    and tuples/lists. Non-tensor values pass through unchanged.
    """
    if isinstance(batch, dict):
        out = {}
        for k, v in batch.items():
            keep = k in keep_dtype_keys
            if torch.is_tensor(v):
                out[k] = _cast_tensor_to_device(v, device, model_dtype, keep)
            elif isinstance(v, dict):
                out[k] = _cast_batch(v, device, model_dtype, keep_dtype_keys)
            elif isinstance(v, (list, tuple)):
                cast = [
                    _cast_tensor_to_device(x, device, model_dtype, keep)
                    if torch.is_tensor(x) else x
                    for x in v
                ]
                out[k] = type(v)(cast)
            else:
                out[k] = v
        return out
    if torch.is_tensor(batch):
        return _cast_tensor_to_device(batch, device, model_dtype, keep_orig=False)
    return batch


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer,
        cfg: TrainerConfig,
        accelerator=None,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.cfg = cfg
        if accelerator is None:
            from accelerate import Accelerator
            from vla_project.training.accelerate_utils import default_ddp_kwargs_handlers

            accelerator = Accelerator(kwargs_handlers=default_ddp_kwargs_handlers())
        self.accelerator = accelerator

    def _save(
        self,
        step: int,
        save_cfg: Any,
        save_norm_stats: Optional[Dict[str, Any]],
        save_tokenizer_settings: Optional[Dict[str, Any]],
    ) -> None:
        if self.cfg.save_dir is None:
            return
        # Sync all ranks before write so the on-disk checkpoint reflects the
        # same step on every rank, then have rank 0 do the actual write.
        if hasattr(self.accelerator, "wait_for_everyone"):
            self.accelerator.wait_for_everyone()
        is_main = getattr(self.accelerator, "is_main_process", True)
        if not is_main:
            return
        unwrapped = self.accelerator.unwrap_model(self.model)
        out = Path(self.cfg.save_dir) / f"step_{step}"
        save_checkpoint(
            out,
            unwrapped,
            step=step,
            cfg=save_cfg if save_cfg is not None else {},
            norm_stats=save_norm_stats,
            optimizer=self.optimizer,
            tokenizer_settings=save_tokenizer_settings,
        )

    def fit(
        self,
        dataloader: Iterable,
        save_cfg: Any = None,
        save_norm_stats: Optional[Dict[str, Any]] = None,
        save_tokenizer_settings: Optional[Dict[str, Any]] = None,
        initial_step: int = 0,
    ) -> List[float]:
        """Train until step counter reaches ``max_steps``.

        Args:
            dataloader: any iterable of batch dicts.
            save_cfg / save_norm_stats / save_tokenizer_settings: optional
                metadata bundled into checkpoints when ``cfg.save_dir`` is set.
            initial_step: starting step for the counter (default 0). Pass the
                ckpt's step when resuming so the LR scheduler (driven by step)
                continues from where it left off. The diagnostic_first_n_batches
                window only fires when initial_step == 0.
        """
        self.model.train()
        # Accelerator.prepare wraps model/optimizer/dataloader for the active
        # backend (DDP under `accelerate launch`, plain torch otherwise).
        self.model, self.optimizer, dataloader = self.accelerator.prepare(
            self.model, self.optimizer, dataloader
        )

        # Inspect the (possibly wrapped) model for device/dtype. unwrap_model
        # gives the underlying module so we read its real param dtype.
        underlying = self.accelerator.unwrap_model(self.model)
        first_param = next(underlying.parameters())
        device = first_param.device
        model_dtype = first_param.dtype
        accum_steps = max(1, int(self.cfg.gradient_accumulation_steps))

        # Snapshot per-group initial lrs for the scheduler (each group keeps
        # its own coefficient applied via build_optimizer; the scheduler only
        # supplies a shared multiplier in [0, 1]).
        initial_lrs = [g["lr"] for g in self.optimizer.param_groups]
        group_names = [g.get("name", f"group_{i}") for i, g in enumerate(self.optimizer.param_groups)]
        scheduler_active = (
            self.cfg.warmup_steps > 0
            or self.cfg.min_lr_ratio < 1.0
            or self.cfg.freeze_steps > 0
        )
        if scheduler_active:
            from vla_project.training.schedulers import linear_warmup_cosine

        losses: List[float] = []
        nan_skip_count = 0
        step = int(initial_step)
        if step > 0 and self.accelerator.is_main_process:
            print(f"[train] resuming step counter from {step} (max_steps={self.cfg.max_steps})", flush=True)
        last_t = time.perf_counter()
        # Exponential moving average of step time for ETA (smooths out the
        # first JIT-warmed steps and noisy data-loading spikes).
        ema_step_s: Optional[float] = None
        ema_alpha = 0.1
        accum_i = 0
        accum_loss_sum = 0.0
        self.optimizer.zero_grad()
        # Per-domain sample counts accumulated over the first
        # diagnostic_first_n_batches batches (B10). Gives a quick view of
        # whether the WeightedMultiDataset draw distribution matches expected
        # uniform / weighted ratios. Skipped on resume (initial_step > 0) —
        # the diagnostic is for catching first-launch data issues, not mid-run.
        diagnostic_n = int(self.cfg.diagnostic_first_n_batches) if step == 0 else 0
        domain_counts: Dict[int, int] = {}
        diag_batches_seen = 0

        while step < self.cfg.max_steps:
            for batch in dataloader:
                # ----- B10 first-N-batches diagnostic --------------------
                if diag_batches_seen < diagnostic_n and isinstance(batch, dict):
                    diag_batches_seen += 1
                    diag_payload: Dict[str, float] = {}
                    if "domain_id" in batch and torch.is_tensor(batch["domain_id"]):
                        ids = batch["domain_id"].detach().cpu().tolist()
                        for i in ids:
                            domain_counts[int(i)] = domain_counts.get(int(i), 0) + 1
                    if "wrist_mask" in batch and torch.is_tensor(batch["wrist_mask"]):
                        wm = batch["wrist_mask"].detach()
                        if wm.numel() > 0:
                            diag_payload["diag/wrist_present_rate"] = float(wm.float().mean().item())
                    if "target_action" in batch and torch.is_tensor(batch["target_action"]):
                        ta = batch["target_action"].detach().float()
                        diag_payload["diag/action_min"] = float(ta.min().item())
                        diag_payload["diag/action_max"] = float(ta.max().item())
                        diag_payload["diag/action_abs_mean"] = float(ta.abs().mean().item())
                    if "proprio" in batch and torch.is_tensor(batch["proprio"]):
                        pp = batch["proprio"].detach().float()
                        diag_payload["diag/proprio_min"] = float(pp.min().item())
                        diag_payload["diag/proprio_max"] = float(pp.max().item())
                    if hasattr(self.accelerator, "log") and diag_payload:
                        # step here is pre-increment; log under the upcoming step
                        self.accelerator.log(diag_payload, step=step + 1)
                    if diag_batches_seen == diagnostic_n:
                        # End-of-window summary: per-domain count distribution
                        # (single emit, attached to the upcoming step).
                        if domain_counts and hasattr(self.accelerator, "log"):
                            total = sum(domain_counts.values())
                            domain_summary = {
                                f"diag/domain_share/{did}": cnt / total
                                for did, cnt in sorted(domain_counts.items())
                            }
                            self.accelerator.log(domain_summary, step=step + 1)
                # --------------------------------------------------------
                batch = _cast_batch(batch, device, model_dtype, self.cfg.keep_dtype_keys)

                # Apply LR schedule for the upcoming optimizer step. `step` is
                # 0-indexed here (incremented after optimizer.step below).
                # Per-group: groups in schedule_group_names get freeze +
                # warmup + (cosine decay if min_lr_ratio<1). Groups outside
                # the list keep their initial constant lr.
                if scheduler_active:
                    for g, init, name in zip(
                        self.optimizer.param_groups, initial_lrs, group_names
                    ):
                        if name not in self.cfg.schedule_group_names:
                            g["lr"] = init  # constant
                            continue
                        eff_freeze = (
                            self.cfg.freeze_steps
                            if name in self.cfg.freeze_group_names
                            else 0
                        )
                        mul = linear_warmup_cosine(
                            step,
                            freeze_steps=eff_freeze,
                            warmup_steps=self.cfg.warmup_steps,
                            total_steps=self.cfg.max_steps,
                            base_lr=1.0,
                            min_lr_ratio=self.cfg.min_lr_ratio,
                        )
                        g["lr"] = init * mul

                _, loss = self.model(batch)
                # NaN guard: a non-finite forward loss propagates through
                # backward → clip_grad_norm_ (norm of NaN is NaN) →
                # optimizer.step (param − lr × NaN = NaN) and poisons every
                # trainable param. No recovery without rewinding to a ckpt,
                # so discard the whole accumulation window when this happens.
                #
                # The check must be DDP-synchronized: rank-local randomness
                # (wrist_view_dropout, dataloader sharding) means one rank
                # can hit non-finite while others stay finite. Skipping on a
                # strict subset of ranks desyncs the next backward/clip/step
                # NCCL collectives, eventually triggering the watchdog
                # timeout (1800 s) and a SIGABRT. Re-applies the gather
                # logic from 1a3db44 (reverted in 0def2dc as part of an
                # unrelated data-loader rollback). v38 nb35 bs=8 hit the
                # un-synced version at step 826.
                loss_finite_local = torch.isfinite(loss).to(torch.uint8).view(1)
                loss_finite_all = self.accelerator.gather(loss_finite_local)
                if not bool(loss_finite_all.all().item()):
                    nan_skip_count += 1
                    self.optimizer.zero_grad()
                    accum_i = 0
                    accum_loss_sum = 0.0
                    if self.accelerator.is_main_process:
                        print(
                            f"[WARN] step {step}: non-finite forward loss on "
                            f"some rank (local={float(loss.detach()):.6g}); "
                            f"skipping accumulation (total skipped: {nan_skip_count})",
                            flush=True,
                        )
                    continue
                # Keep optimizer-step gradients equivalent to a large batch by
                # averaging microbatch losses before backward. Reporting below
                # logs the mean *raw* loss over the accumulation window.
                self.accelerator.backward(loss / accum_steps)
                accum_loss_sum += float(loss.detach().item())
                accum_i += 1
                if accum_i < accum_steps:
                    continue
                # Capture pre-clip total grad norm for diagnostics. Even with
                # finite forward loss, bf16 backward can still overflow, so
                # also guard the optimizer step against non-finite norm.
                # DDP backward allreduces gradients in principle, but the
                # post-allreduce float() round-trip + per-rank optimizer
                # state still leaves room for divergence under accumulation,
                # so we gather the finite-flag here too. v38 nb35 bs=8
                # surfaced this path as the visible failure (step 826
                # "non-finite grad_norm (inf)" → 30 min later watchdog SIGABRT).
                grad_norm = self.accelerator.clip_grad_norm_(
                    self.model.parameters(), self.cfg.grad_clip_norm
                )
                grad_norm_val = float(grad_norm) if grad_norm is not None else 0.0
                grad_norm_finite_local = torch.tensor(
                    [int(math.isfinite(grad_norm_val))],
                    dtype=torch.uint8,
                    device=self.accelerator.device,
                )
                grad_norm_finite_all = self.accelerator.gather(grad_norm_finite_local)
                if not bool(grad_norm_finite_all.all().item()):
                    nan_skip_count += 1
                    self.optimizer.zero_grad()
                    accum_i = 0
                    accum_loss_sum = 0.0
                    if self.accelerator.is_main_process:
                        print(
                            f"[WARN] step {step}: non-finite grad_norm on "
                            f"some rank (local={grad_norm_val}); "
                            f"skipping optimizer.step "
                            f"(total skipped: {nan_skip_count})",
                            flush=True,
                        )
                    continue
                self.optimizer.step()
                self.optimizer.zero_grad()

                # Cross-rank average for reporting; single-GPU is a no-op.
                report_loss = loss.detach().new_tensor(accum_loss_sum / accum_steps)
                gathered = self.accelerator.gather_for_metrics(report_loss)
                loss_val = float(gathered.mean().item())
                losses.append(loss_val)
                accum_i = 0
                accum_loss_sum = 0.0

                step += 1
                now = time.perf_counter()
                step_time_s = now - last_t
                last_t = now
                ema_step_s = (
                    step_time_s if ema_step_s is None
                    else ema_alpha * step_time_s + (1 - ema_alpha) * ema_step_s
                )
                eta_s = ema_step_s * max(0, self.cfg.max_steps - step)
                progress_pct = 100.0 * step / max(1, self.cfg.max_steps)
                # accelerator.log() is a no-op when no tracker was registered
                # (e.g., default Accelerator()); when log_with='wandb' was set
                # at construction, this routes to wandb.log(..., step=step).
                # Logged after step += 1 so the first call is step=1.
                if hasattr(self.accelerator, "log"):
                    payload = {
                        "train/loss":         loss_val,
                        "train/grad_norm":    grad_norm_val,
                        "train/nan_skip_count": float(nan_skip_count),
                        "train/step_time_ms": step_time_s * 1000.0,
                        "train/progress_pct": progress_pct,
                        "train/eta_s":        eta_s,
                        "train/gradient_accumulation_steps": accum_steps,
                    }
                    # Per-group lr is informative when scheduling is active and
                    # also useful as a sanity check for the per-group coefs.
                    for name, g in zip(group_names, self.optimizer.param_groups):
                        payload[f"train/lr/{name}"] = g["lr"]
                    # EE6D (or any other model that exposes per-channel loss
                    # components) attaches them under `_last_loss_info` so we
                    # can plot pos / rot / grip separately without changing
                    # the forward() signature.
                    extra = getattr(underlying, "_last_loss_info", None)
                    if extra:
                        for k, v in extra.items():
                            payload[k] = float(v.item()) if torch.is_tensor(v) else float(v)
                    self.accelerator.log(payload, step=step)

                # Periodic save (only when both save_every and save_dir set).
                if (
                    self.cfg.save_dir is not None
                    and self.cfg.save_every is not None
                    and step > 0
                    and step % self.cfg.save_every == 0
                ):
                    self._save(step, save_cfg, save_norm_stats, save_tokenizer_settings)

                if step >= self.cfg.max_steps:
                    break

        # Final save at end of fit. Skip when the most recent periodic save
        # already wrote this step (max_steps a multiple of save_every) so we
        # don't redo identical work.
        periodic_just_fired = (
            self.cfg.save_every is not None
            and step > 0
            and step % self.cfg.save_every == 0
        )
        if self.cfg.save_dir is not None and not periodic_just_fired:
            self._save(step, save_cfg, save_norm_stats, save_tokenizer_settings)
        # Close any open tracker run (no-op if none was initialized).
        if hasattr(self.accelerator, "end_training"):
            self.accelerator.end_training()
        return losses
