"""Trainer uses an Accelerator for backward + gather.

Stub-Accelerator-based test — verifies Trainer plays nicely with the
Accelerator interface even on CPU-only / single-process runs.
"""
from typing import Any, Iterable

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from vla_project.training.trainer import Trainer, TrainerConfig


class _Toy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(3, 1)

    def forward(self, batch: dict):
        x = batch["x"]
        y = batch["y"]
        pred = self.fc(x)
        loss = (pred - y).abs().mean()
        return pred, loss


class _ToyDS(Dataset):
    def __len__(self) -> int:
        return 8

    def __getitem__(self, idx: int) -> dict:
        return {
            "x": torch.randn(3),
            "y": torch.randn(1),
        }


def _collate(samples):
    return {k: torch.stack([s[k] for s in samples]) for k in samples[0]}


class _StubAccelerator:
    """Mimics the subset of accelerate.Accelerator that Trainer touches."""
    def __init__(self) -> None:
        self.backward_calls = 0
        self.prepare_calls = 0

    def prepare(self, *args):
        self.prepare_calls += 1
        return args if len(args) > 1 else args[0]

    def backward(self, loss: torch.Tensor) -> None:
        self.backward_calls += 1
        loss.backward()

    def gather_for_metrics(self, t: torch.Tensor) -> torch.Tensor:
        return t.unsqueeze(0)

    def unwrap_model(self, model: nn.Module) -> nn.Module:
        return model


def test_trainer_uses_accelerator_for_backward() -> None:
    model = _Toy()
    opt = torch.optim.SGD(model.parameters(), lr=1e-3)
    ds = _ToyDS()
    dl = DataLoader(ds, batch_size=2, collate_fn=_collate)
    acc = _StubAccelerator()
    trainer = Trainer(model, opt, TrainerConfig(max_steps=3), accelerator=acc)
    losses = trainer.fit(dl)
    assert len(losses) == 3
    assert all(isinstance(l, float) for l in losses)
    assert acc.backward_calls == 3
    assert acc.prepare_calls >= 1


def test_trainer_default_accelerator_still_runs() -> None:
    """When no Accelerator is supplied, Trainer constructs a real one."""
    model = _Toy()
    opt = torch.optim.SGD(model.parameters(), lr=1e-3)
    dl = DataLoader(_ToyDS(), batch_size=2, collate_fn=_collate)
    trainer = Trainer(model, opt, TrainerConfig(max_steps=2))
    losses = trainer.fit(dl)
    assert len(losses) == 2
