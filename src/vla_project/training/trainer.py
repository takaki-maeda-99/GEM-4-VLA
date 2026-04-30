"""Minimal Trainer with Accelerator-driven backward / loss gather.

Single-GPU `python scripts/train.py ...` and multi-GPU `accelerate launch
... scripts/train.py ...` use the same code path. Accelerator()'s no-arg
constructor reads env vars set by `accelerate launch`; in single-process
mode it is a near-no-op.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import torch
import torch.nn as nn

from vla_project.training.checkpoint import save_checkpoint


@dataclass
class TrainerConfig:
    max_steps: int = 100
    log_every: int = 10
    grad_clip_norm: float = 1.0
    save_every: Optional[int] = None  # save every N steps; None disables periodic
    save_dir: Optional[str] = None    # parent dir for step_<N>/ checkpoints


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
            accelerator = Accelerator()
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
    ) -> List[float]:
        """Train for exactly ``max_steps`` optimizer steps.

        Args:
            dataloader: any iterable of batch dicts.
            save_cfg / save_norm_stats / save_tokenizer_settings: optional
                metadata bundled into checkpoints when ``cfg.save_dir`` is set.
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

        losses: List[float] = []
        step = 0
        while step < self.cfg.max_steps:
            for batch in dataloader:
                cast_batch = {}
                for k, v in batch.items():
                    if not torch.is_tensor(v):
                        cast_batch[k] = v
                        continue
                    v = v.to(device)
                    if v.is_floating_point():
                        v = v.to(model_dtype)
                    cast_batch[k] = v
                batch = cast_batch

                self.optimizer.zero_grad()
                _, loss = self.model(batch)
                self.accelerator.backward(loss)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.cfg.grad_clip_norm
                )
                self.optimizer.step()

                # Cross-rank average for reporting; single-GPU is a no-op.
                gathered = self.accelerator.gather_for_metrics(loss.detach())
                losses.append(float(gathered.mean().item()))

                step += 1

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

        # Final save at end of fit (always when save_dir is set).
        if self.cfg.save_dir is not None:
            self._save(step, save_cfg, save_norm_stats, save_tokenizer_settings)
        return losses
