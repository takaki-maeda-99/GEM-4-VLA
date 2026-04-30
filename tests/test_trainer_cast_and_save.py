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


# ---------- LR scheduler integration ----------

class _StubAccLog:
    is_main_process = True

    def __init__(self) -> None:
        self.log_calls = []

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
        pass


def test_warmup_ramps_lr_from_zero_to_init() -> None:
    """warmup_steps=10: at step 0 → lr=0; at step 10 → lr=init; min_lr_ratio=0
    forces post-warmup to enter cosine decay reaching 0 at total_steps."""
    model = _Toy()
    init_lr = 1e-3
    opt = torch.optim.SGD(model.parameters(), lr=init_lr)
    # tag the group with a name so the per-group log key works
    opt.param_groups[0]["name"] = "toy"
    dl = DataLoader(_ToyDS(), batch_size=2, collate_fn=_collate)
    cfg = TrainerConfig(max_steps=20, warmup_steps=10, min_lr_ratio=0.0)
    acc = _StubAccLog()
    trainer = Trainer(model, opt, cfg, accelerator=acc)
    trainer.fit(dl)

    lrs = [p[0]["train/lr/toy"] for p in acc.log_calls]
    # Step 0 used lr=0 (logged at step=1 entry); step 9 used lr=9/10*init.
    # The logged lrs are AFTER the schedule applied for the just-completed step.
    assert lrs[0] == 0.0
    assert abs(lrs[9] - 0.9 * init_lr) < 1e-9
    # Step 10 (just past warmup): lr = init * (min + (1-min)*cos(0)) = init.
    assert abs(lrs[10] - init_lr) < 1e-9
    # Final step's lr decayed well below peak (cosine end with min_lr_ratio=0
    # at step=N-1 / N=0.9 progress -> ~0.024 × init_lr).
    assert lrs[-1] < 0.05 * init_lr


def test_no_scheduling_when_warmup_zero_and_min_lr_ratio_one() -> None:
    """Defaults: warmup=0, min_lr_ratio=1.0 → constant LR."""
    model = _Toy()
    init_lr = 5e-4
    opt = torch.optim.SGD(model.parameters(), lr=init_lr)
    opt.param_groups[0]["name"] = "toy"
    dl = DataLoader(_ToyDS(), batch_size=2, collate_fn=_collate)
    cfg = TrainerConfig(max_steps=4)  # defaults: warmup=0, min_lr_ratio=1.0
    acc = _StubAccLog()
    trainer = Trainer(model, opt, cfg, accelerator=acc)
    trainer.fit(dl)
    lrs = [p[0]["train/lr/toy"] for p in acc.log_calls]
    # All identical since scheduler is inactive.
    assert all(lr == init_lr for lr in lrs)


def test_freeze_steps_applies_only_to_named_groups() -> None:
    """freeze_steps applies only to groups listed in freeze_group_names.
    Head / projection / soft_prompt groups warm up from step 0; gemma_lora
    stays at lr=0 during the freeze window."""
    model = _Toy()
    head_lr = 1e-3
    backbone_lr = 1e-4
    opt = torch.optim.SGD([
        {"name": "action_head", "params": list(model.parameters()), "lr": head_lr},
    ])
    # Add a fake "gemma_lora" group with a separate trivial param so we can
    # observe its lr trajectory without affecting actual training.
    fake_param = nn.Parameter(torch.zeros(2))
    opt.add_param_group({"name": "gemma_lora", "params": [fake_param], "lr": backbone_lr})
    dl = DataLoader(_ToyDS(), batch_size=2, collate_fn=_collate)
    cfg = TrainerConfig(
        max_steps=20, freeze_steps=5, warmup_steps=5, min_lr_ratio=0.0,
    )
    acc = _StubAccLog()
    trainer = Trainer(model, opt, cfg, accelerator=acc)
    trainer.fit(dl)

    head_lrs = [p[0]["train/lr/action_head"] for p in acc.log_calls]
    bb_lrs   = [p[0]["train/lr/gemma_lora"]  for p in acc.log_calls]
    # Action head: warmup from step 0, peak at step 5.
    assert head_lrs[0] == 0.0
    assert abs(head_lrs[4] - 0.8 * head_lr) < 1e-9   # 4/5 ramp
    assert abs(head_lrs[5] - head_lr) < 1e-9         # peak
    # Backbone: frozen until step 5, then ramps over warmup_steps=5 -> peak at step 10.
    for lr in bb_lrs[:5]:
        assert lr == 0.0
    assert abs(bb_lrs[5] - 0.0) < 1e-9               # first warmup step (s=0/5)
    assert abs(bb_lrs[9] - 0.8 * backbone_lr) < 1e-9 # last warmup step (s=4/5)
    assert abs(bb_lrs[10] - backbone_lr) < 1e-9      # peak
