# Accelerate Distributed Launch (Plan 10 / 10)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Enable multi-GPU training via `accelerate launch`. Wire `accelerate.Accelerator` into `Trainer.fit()` such that single-GPU runs (`python scripts/train.py ...`) keep working unchanged AND distributed runs (`accelerate launch --num_processes 2 --gpu_ids 3,4 scripts/train.py ...`) split work across GPUs. End-to-end smoke verifies both modes on dl40 with the existing synthetic config.

**Architecture:** `accelerate.Accelerator()` is a near-no-op when not launched in a distributed context. We make `Trainer` consult it always:

1. `Trainer.fit(dataloader)` constructs (or accepts) an `Accelerator` instance.
2. `accelerator.prepare(model, optimizer, dataloader)` wraps each for the active backend (DDP under `accelerate launch`, plain torch otherwise). For our `IterableDataset` children, `prepare` is essentially a passthrough.
3. `accelerator.backward(loss)` replaces `loss.backward()`. In single-GPU mode it is identical; in DDP it triggers gradient sync.
4. Loss reporting collects across ranks: `loss_local = loss.detach()`; `loss_all = accelerator.gather_for_metrics(loss_local)`; reported value = `loss_all.mean().item()`.

The device/dtype hand-casting in `Trainer.fit` stays as-is — `prepare()` does not auto-cast our custom batch dicts (which contain non-tensor `domain_id`-style values), and the existing logic works in both modes.

`scripts/train.py` does **not** need changes — `accelerate launch` injects env vars (`LOCAL_RANK`, `WORLD_SIZE`, etc.) that `Accelerator()` reads automatically.

**Tech Stack:** `accelerate>=1.0` (already a dep since Plan 1).

**Repo references:**
- `src/vla_project/training/trainer.py` — single point of mutation. ~30 lines today.
- `src/vla_project/training/checkpoint.py` (Plan 4) — independent; `accelerator.unwrap_model()` is the recommended way to get a state_dict on rank 0 in distributed mode, but checkpoint wiring is out of scope here (Plan 4 isn't called from Trainer yet).
- `accelerate` docs: https://huggingface.co/docs/accelerate

**Hard constraints from CLAUDE.md:**
- Existing tests must not regress.
- Single-GPU `python scripts/train.py configs/train/smoke.yaml` continues to work and produce two finite losses.
- Multi-GPU smoke (2 ranks) produces finite losses and prints them once per rank-0.

---

## File Structure

**Modify:**
- `src/vla_project/training/trainer.py` (use Accelerator)
- `src/vla_project/training/__init__.py` (no changes; existing re-exports stay)
- `tests/test_trainer_one_step.py` (verify behaviour preserved on single-GPU CPU path)

**Create:**
- `tests/test_trainer_accelerator.py` (passes a no-op Accelerator stub; verifies the call paths)
- `configs/accelerate/dl40_2gpu.yaml` (accelerate config preset for dl40's free 40GB GPUs)

**Do not modify:** `models/`, `policies/`, `robots/`, `evaluation/`, `data/`, `scripts/train.py`.

---

## Task 1: Wire `Accelerator` into `Trainer`

**Files:**
- Modify: `src/vla_project/training/trainer.py`
- Create: `tests/test_trainer_accelerator.py`

- [ ] **Step 1: Read existing Trainer**

```bash
cat src/vla_project/training/trainer.py
```

Note current behaviour:
- `Trainer(model, optimizer, cfg)` — three positional args; cfg has `max_steps`, `log_every`, `grad_clip_norm`.
- `fit(dataloader)` returns `List[float]` of per-step losses.
- Per-batch device/dtype casting is done inline.

- [ ] **Step 2: Write failing test**

`tests/test_trainer_accelerator.py`:

```python
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
    """Mimics the subset of accelerate.Accelerator that Trainer touches.

    `prepare` returns its inputs unchanged; `backward` calls `.backward()`
    directly; `gather_for_metrics` returns the input tensor unwrapped to
    a single-element batch dimension.
    """
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
```

- [ ] **Step 3: Run, expect FAIL**

```bash
PYTHONPATH="" uv run pytest tests/test_trainer_accelerator.py -v
```

Expected: `TypeError` on `Trainer(... accelerator=acc)` (the kwarg doesn't exist yet).

- [ ] **Step 4: Implement**

Replace `src/vla_project/training/trainer.py` with:

```python
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

        # Inspect the (possibly wrapped) model for device/dtype. ``unwrap_model``
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
```

- [ ] **Step 5: Run, expect PASS**

```bash
PYTHONPATH="" uv run pytest tests/test_trainer_accelerator.py tests/test_trainer_one_step.py -v
PYTHONPATH="" uv run pytest -q
```

Expected: 2 new tests pass, existing trainer test still passes; full suite green (110 + 2 = 112).

If `tests/test_trainer_one_step.py` regresses because the existing test passed `Trainer(model, opt, cfg)` (3 args) and our new `accelerator=None` default works, no change to that test should be needed. Verify.

- [ ] **Step 6: Quick single-GPU smoke**

```bash
CUDA_VISIBLE_DEVICES=3 PYTHONPATH="" uv run python scripts/train.py configs/train/smoke.yaml 2>&1 | tail -3
```

Expected: `[train] losses=[<f1>, <f2>]` — two finite floats, same scale as before.

- [ ] **Step 7: Commit**

```bash
git add src/vla_project/training/trainer.py tests/test_trainer_accelerator.py
git commit -m "feat(training): Trainer.fit uses Accelerator for backward + gather"
```

---

## Task 2: Multi-GPU smoke + accelerate config

**Files:**
- Create: `configs/accelerate/dl40_2gpu.yaml`

- [ ] **Step 1: Inspect available accelerate config**

```bash
ls ~/.cache/huggingface/accelerate/ 2>&1 | head
PYTHONPATH="" uv run python -c "from accelerate.commands.config import default_config_file; print(default_config_file())"
```

Either we already have a default config or we don't. We add an explicit per-host preset to avoid relying on global state.

- [ ] **Step 2: Write the preset**

`configs/accelerate/dl40_2gpu.yaml`:

```yaml
# Accelerate launch preset for dl40: 2x A100 40GB on GPUs 3 + 4 (the
# typically-idle pair). Override with --num_processes / --gpu_ids on the
# command line if needed.
compute_environment: LOCAL_MACHINE
distributed_type: MULTI_GPU
downcast_bf16: 'no'
gpu_ids: 3,4
machine_rank: 0
main_training_function: main
mixed_precision: bf16
num_machines: 1
num_processes: 2
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
```

- [ ] **Step 3: Run multi-GPU smoke**

```bash
cd /misc/dl00/takaki/X-VLA-Adapter
PYTHONPATH="" timeout 600 \
    uv run accelerate launch --config_file configs/accelerate/dl40_2gpu.yaml \
    scripts/train.py configs/train/smoke.yaml 2>&1 | tee /tmp/smoke_2gpu.log | tail -10
```

Expected: `[train] device=cuda dtype=...` lines from BOTH ranks (or only rank 0 if accelerate suppresses), then a single `[train] losses=[<f1>, <f2>]` from rank 0. Both losses finite. No NCCL hang.

If the smoke gets stuck at NCCL init (common on shared hosts), STOP and report. Possible causes: GPU 3 or 4 unexpectedly in use, or another rank already binding the same port. Check:

```bash
nvidia-smi --query-gpu=index,memory.free --format=csv,noheader
```

Pick a different `gpu_ids` pair (e.g. `4,7`) and adjust the config.

- [ ] **Step 4: Commit + push + PR**

```bash
git add configs/accelerate/dl40_2gpu.yaml
git commit -m "feat(configs): accelerate 2x A100 launch preset for dl40"
git push -u origin feat/accelerate-distributed
```

PR base: `feat/wrist-pool-huber` (rebase to `main` once Plans 1-9 merge).
Title: `feat(training): Accelerate-based distributed launch`.
Body should include:
- Test count delta (110 → 112 expected).
- Single-GPU `losses=[...]` confirming backward-compat.
- Multi-GPU `losses=[...]` from the 2-GPU run.
- Note: `prepare(IterableDataset)` is a near-passthrough; per-rank dataset duplication still applies. Multi-rank weighted-mixer determinism is a follow-up.

---

## Done criteria

- [ ] `uv run pytest -q` green (112 expected).
- [ ] `python scripts/train.py configs/train/smoke.yaml` (single-GPU) prints two finite losses.
- [ ] `accelerate launch --config_file configs/accelerate/dl40_2gpu.yaml scripts/train.py configs/train/smoke.yaml` prints two finite losses (rank-0 average).
- [ ] No edits to `models/`, `policies/`, `robots/`, `evaluation/`, `data/`, `scripts/train.py`.

## Out of scope (follow-ups)

- `accelerator.save_state` / `load_state` integration with Plan 4's checkpoint format (Plan 4's checkpoint uses raw `state_dict`; Accelerate has its own state-dir layout for full restore including RNG seeds).
- Per-rank dataset sharding for `LeRobotLiberoDataset` / `WeightedMultiDataset`. Currently each rank iterates the full dataset; for true sharded training a `worker_info`-aware splitter is needed (see TODO in `WeightedMultiDataset` from Plan 3).
- DeepSpeed / FSDP configurations (Accelerate supports them; we picked vanilla DDP for the smoke).
- Mixed-precision policy beyond `bf16` (cf. `downcast_bf16`).
