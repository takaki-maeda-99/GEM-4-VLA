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
        """
        self.model.train()
        losses: List[float] = []
        step = 0
        while step < self.cfg.max_steps:
            for batch in dataloader:
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
