"""Trainer batch casting (keep_dtype_keys + nested) + dedup final save."""
from pathlib import Path
from typing import Any, Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from vla_project.training.trainer import (
    Trainer,
    TrainerConfig,
    _cast_batch,
)


# ---------- _cast_batch unit tests (no Trainer needed) ----------

def test_cast_batch_keep_dtype_for_target_action() -> None:
    """target_action stays float32 even when model_dtype is bf16."""
    batch = {
        "scene_image":   torch.randn(2, 3, 4, 4, dtype=torch.float32),
        "target_action": torch.randn(2, 8, 7, dtype=torch.float32),
        "action_mask":   torch.ones(2, 8, dtype=torch.bool),
        "domain_id":     torch.zeros(2, dtype=torch.long),
    }
    out = _cast_batch(batch, device="cpu", model_dtype=torch.bfloat16,
                      keep_dtype_keys=("target_action",))
    assert out["scene_image"].dtype == torch.bfloat16
    assert out["target_action"].dtype == torch.float32  # KEPT
    assert out["action_mask"].dtype == torch.bool        # bool unchanged anyway
    assert out["domain_id"].dtype == torch.long          # long unchanged


def test_cast_batch_handles_nested_dict() -> None:
    """{"obs": {"image": ..., "proprio": ...}, "action": ...} all reach device."""
    batch = {
        "obs": {
            "image":   torch.randn(2, 3, 4, 4, dtype=torch.float32),
            "proprio": torch.randn(2, 8, dtype=torch.float32),
        },
        "action": torch.randn(2, 8, 7, dtype=torch.float32),
    }
    out = _cast_batch(batch, device="cpu", model_dtype=torch.bfloat16,
                      keep_dtype_keys=("target_action",))
    assert out["obs"]["image"].dtype == torch.bfloat16
    assert out["obs"]["proprio"].dtype == torch.bfloat16
    # 'action' is NOT in keep list (the project key is 'target_action')
    assert out["action"].dtype == torch.bfloat16


def test_cast_batch_handles_lists_and_tuples() -> None:
    batch = {
        "frames": [torch.randn(3, 4, dtype=torch.float32) for _ in range(2)],
        "scalars": (torch.tensor(1.0), torch.tensor(2.0)),
    }
    out = _cast_batch(batch, device="cpu", model_dtype=torch.bfloat16, keep_dtype_keys=())
    assert isinstance(out["frames"], list)
    assert all(t.dtype == torch.bfloat16 for t in out["frames"])
    assert isinstance(out["scalars"], tuple)
    assert all(t.dtype == torch.bfloat16 for t in out["scalars"])


def test_cast_batch_passthrough_non_tensors() -> None:
    batch = {"img": torch.randn(2, 3), "task": "pick up", "id": 42}
    out = _cast_batch(batch, device="cpu", model_dtype=torch.bfloat16, keep_dtype_keys=())
    assert out["task"] == "pick up"
    assert out["id"] == 42
    assert out["img"].dtype == torch.bfloat16


# ---------- dedup-save integration ----------

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


class _StubAcc:
    is_main_process = True

    def __init__(self) -> None:
        self.save_count = 0

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


def test_no_double_save_when_max_steps_aligns_with_save_every(tmp_path: Path) -> None:
    """When max_steps % save_every == 0, the periodic save at step==max_steps
    is the same as the final save → final should be skipped."""
    model = _Toy()
    opt = torch.optim.SGD(model.parameters(), lr=1e-3)
    dl = DataLoader(_ToyDS(), batch_size=2, collate_fn=_collate)
    cfg = TrainerConfig(max_steps=4, save_every=2, save_dir=str(tmp_path))
    trainer = Trainer(model, opt, cfg, accelerator=_StubAcc())
    trainer.fit(dl, save_cfg={})

    # Step 2 + step 4 (periodic) should exist; final save dedup skipped.
    dirs = sorted(p.name for p in tmp_path.iterdir())
    assert dirs == ["step_2", "step_4"]
    # Inspect mtimes — step_4 should have been written once, not twice.
    # (Rewriting would update mtime; we accept that's hard to verify cheaply.
    # The dirs check is the load-bearing assertion.)


def test_final_save_still_fires_when_unaligned(tmp_path: Path) -> None:
    """When max_steps is NOT a multiple of save_every, final save must run."""
    model = _Toy()
    opt = torch.optim.SGD(model.parameters(), lr=1e-3)
    dl = DataLoader(_ToyDS(), batch_size=2, collate_fn=_collate)
    cfg = TrainerConfig(max_steps=5, save_every=2, save_dir=str(tmp_path))
    trainer = Trainer(model, opt, cfg, accelerator=_StubAcc())
    trainer.fit(dl, save_cfg={})

    # Periodic at 2, 4; final at 5.
    dirs = sorted(p.name for p in tmp_path.iterdir())
    assert dirs == ["step_2", "step_4", "step_5"]
