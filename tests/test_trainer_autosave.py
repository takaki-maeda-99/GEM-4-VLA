"""Trainer.fit auto-saves checkpoints at save_every and at end."""
from pathlib import Path

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
        return 16

    def __getitem__(self, idx: int) -> dict:
        return {"x": torch.randn(3), "y": torch.randn(1)}


def _collate(samples):
    return {k: torch.stack([s[k] for s in samples]) for k in samples[0]}


class _StubAccelerator:
    is_main_process = True

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


def _build_trainer(tmp_path: Path, save_every, max_steps: int = 4):
    model = _Toy()
    opt = torch.optim.SGD(model.parameters(), lr=1e-3)
    dl = DataLoader(_ToyDS(), batch_size=2, collate_fn=_collate)
    cfg = TrainerConfig(max_steps=max_steps, save_every=save_every, save_dir=str(tmp_path))
    return Trainer(model, opt, cfg, accelerator=_StubAccelerator()), dl


def test_periodic_save_creates_step_dirs(tmp_path: Path) -> None:
    trainer, dl = _build_trainer(tmp_path, save_every=2, max_steps=4)
    trainer.fit(dl, save_cfg={"smoke": True})
    # Periodic at step 2 + step 4; final at step 4 overwrites step 4 atomically.
    dirs = sorted(p.name for p in tmp_path.iterdir())
    assert dirs == ["step_2", "step_4"]
    assert (tmp_path / "step_2" / "model.pt").is_file()
    assert (tmp_path / "step_4" / "meta.json").is_file()


def test_final_save_only_when_no_save_every(tmp_path: Path) -> None:
    trainer, dl = _build_trainer(tmp_path, save_every=None, max_steps=3)
    trainer.fit(dl, save_cfg={"final_only": True})
    dirs = sorted(p.name for p in tmp_path.iterdir())
    assert dirs == ["step_3"]


def test_no_save_when_save_dir_none(tmp_path: Path) -> None:
    model = _Toy()
    opt = torch.optim.SGD(model.parameters(), lr=1e-3)
    dl = DataLoader(_ToyDS(), batch_size=2, collate_fn=_collate)
    cfg = TrainerConfig(max_steps=2)  # save_dir defaults to None
    trainer = Trainer(model, opt, cfg, accelerator=_StubAccelerator())
    trainer.fit(dl)
    assert sorted(p.name for p in tmp_path.iterdir()) == []


def test_meta_records_norm_stats_and_tokenizer(tmp_path: Path) -> None:
    trainer, dl = _build_trainer(tmp_path, save_every=None, max_steps=2)
    norm_stats = {"libero_test": {"action": {"q01": [0]*7, "q99": [1]*7, "mask": [True]*7}}}
    tok_settings = {"model_name": "google/gemma-4-E2B", "max_len": 50}
    trainer.fit(dl, save_norm_stats=norm_stats, save_tokenizer_settings=tok_settings)
    import json
    meta = json.loads((tmp_path / "step_2" / "meta.json").read_text())
    assert meta["norm_stats"] == norm_stats
    assert meta["tokenizer_settings"] == tok_settings
