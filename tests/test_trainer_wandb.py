"""Trainer.fit logs per-step loss via accelerator.log; end_training is called."""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from vla_project.training.trainer import Trainer, TrainerConfig


class _Toy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(3, 1)

    def forward(self, batch: dict):
        pred = self.fc(batch["x"])
        loss = (pred - batch["y"]).abs().mean()
        return pred, loss


class _ToyDS(Dataset):
    def __len__(self) -> int:
        return 8

    def __getitem__(self, idx: int) -> dict:
        return {"x": torch.randn(3), "y": torch.randn(1)}


def _collate(samples):
    return {k: torch.stack([s[k] for s in samples]) for k in samples[0]}


class _StubAcceleratorWithLog:
    is_main_process = True

    def __init__(self) -> None:
        self.log_calls = []
        self.end_training_calls = 0

    def prepare(self, *args):
        return args if len(args) > 1 else args[0]

    def backward(self, loss):
        loss.backward()

    def clip_grad_norm_(self, params, max_norm):
        return torch.nn.utils.clip_grad_norm_(params, max_norm)

    def gather_for_metrics(self, t):
        return t.unsqueeze(0)

    def unwrap_model(self, m):
        return m

    def wait_for_everyone(self):
        pass

    def log(self, payload, step=None):
        self.log_calls.append((dict(payload), step))

    def end_training(self):
        self.end_training_calls += 1


def test_trainer_logs_each_step() -> None:
    model = _Toy()
    opt = torch.optim.SGD(model.parameters(), lr=1e-3)
    dl = DataLoader(_ToyDS(), batch_size=2, collate_fn=_collate)
    acc = _StubAcceleratorWithLog()
    trainer = Trainer(model, opt, TrainerConfig(max_steps=4), accelerator=acc)
    trainer.fit(dl)
    assert len(acc.log_calls) == 4
    for i, (payload, step) in enumerate(acc.log_calls, start=1):
        assert "train/loss" in payload
        assert isinstance(payload["train/loss"], float)
        assert "train/step_time_ms" in payload
        assert payload["train/step_time_ms"] >= 0.0
        assert "train/progress_pct" in payload
        assert payload["train/progress_pct"] == 25.0 * i  # 4 steps -> 25/50/75/100
        assert "train/eta_s" in payload
        assert payload["train/eta_s"] >= 0.0
        assert step == i
    # ETA should reach 0 on the last step (no remaining steps).
    assert acc.log_calls[-1][0]["train/eta_s"] == 0.0
    assert acc.end_training_calls == 1


def test_trainer_log_call_passes_step_kwarg() -> None:
    """Verify the log() call uses step=N (so wandb x-axis is the train step)."""
    model = _Toy()
    opt = torch.optim.SGD(model.parameters(), lr=1e-3)
    dl = DataLoader(_ToyDS(), batch_size=2, collate_fn=_collate)
    acc = _StubAcceleratorWithLog()
    trainer = Trainer(model, opt, TrainerConfig(max_steps=2), accelerator=acc)
    trainer.fit(dl)
    steps = [s for _, s in acc.log_calls]
    assert steps == [1, 2]
