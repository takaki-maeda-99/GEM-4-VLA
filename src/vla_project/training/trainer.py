from dataclasses import dataclass
from typing import Iterable, List

import torch
import torch.nn as nn


@dataclass
class TrainerConfig:
    max_steps: int = 100
    log_every: int = 10
    grad_clip_norm: float = 1.0


class Trainer:
    def __init__(self, model: nn.Module, optimizer, cfg: TrainerConfig) -> None:
        self.model = model
        self.optimizer = optimizer
        self.cfg = cfg

    def fit(self, dataloader: Iterable) -> List[float]:
        """Train for exactly `max_steps` optimizer steps, re-iterating the
        dataloader as needed. Calling `iter(dataloader)` again starts a new
        epoch (fresh shuffling for map-style datasets, restart for iterable).

        Batch tensors are moved to the model's device before forward.
        """
        self.model.train()
        first_param = next(self.model.parameters())
        device = first_param.device
        model_dtype = first_param.dtype
        losses: List[float] = []
        step = 0
        while step < self.cfg.max_steps:
            for batch in dataloader:
                # Move tensors to device. Float-typed inputs (images, proprio,
                # actions) are also cast to the model's dtype so they line up
                # with parameter dtypes (Linear ops require matching dtypes).
                # Long/bool tensors (input_ids, attention_mask, action_mask,
                # domain_id) are left in their integer dtype.
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
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip_norm)
                self.optimizer.step()
                losses.append(loss.item())
                step += 1
                if step >= self.cfg.max_steps:
                    break
        return losses
