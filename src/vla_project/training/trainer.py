"""Minimal Trainer with Accelerator-driven backward / loss gather.

Single-GPU `python scripts/train.py ...` and multi-GPU `accelerate launch
... scripts/train.py ...` use the same code path. Accelerator()'s no-arg
constructor reads env vars set by `accelerate launch`; in single-process
mode it is a near-no-op.
"""
from dataclasses import dataclass
from typing import Iterable, List, Optional

import torch
import torch.nn as nn


@dataclass
class TrainerConfig:
    max_steps: int = 100
    log_every: int = 10
    grad_clip_norm: float = 1.0


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

    def fit(self, dataloader: Iterable) -> List[float]:
        """Train for exactly ``max_steps`` optimizer steps."""
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
                if step >= self.cfg.max_steps:
                    break
        return losses
