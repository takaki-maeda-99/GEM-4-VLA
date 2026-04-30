# Checkpoint Save/Load Implementation Plan (Plan 4 / 10)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `training/checkpoint.py` with `save_checkpoint(...)` and `load_checkpoint(...)` that produce a self-contained checkpoint directory matching CLAUDE.md's "Experiment Outputs" rules — model weights + resolved config + norm_stats + git commit hash + step counter — and verify a round-trip restores the model to bit-exact equality with the original. No changes to `Trainer` or `scripts/train.py` (callers will wire it later).

**Architecture:** A checkpoint is a directory `checkpoints/step_<N>/` containing:
- `model.pt` — `torch.save(model.state_dict())`.
- `optimizer.pt` — `torch.save(optimizer.state_dict())` (optional; only written when an optimizer is passed to save).
- `meta.json` — JSON with `step`, `git_commit`, `tokenizer_settings`, resolved `cfg` (as a plain dict), and the `norm_stats` payload (a dict shaped like `data/norm_stats/*.json`).

`save_checkpoint(out_dir, model, *, step, cfg, norm_stats=None, optimizer=None, tokenizer_settings=None)` creates the directory atomically (write to a temp dir, then rename) so a partial write cannot leave a half-baked checkpoint. `load_checkpoint(in_dir, model, *, optimizer=None) -> dict` restores `state_dict`s in place and returns the meta dict. Both raise `FileNotFoundError` if the directory is missing/incomplete.

**Tech Stack:** stdlib `json`, `pathlib`, `subprocess` (for `git rev-parse HEAD`), existing `torch.save / torch.load`, `OmegaConf.to_container` for resolving `${oc.env:...}` interpolations to plain values.

**Repo references:**
- `CLAUDE.md` "Experiment Outputs" section — locks the on-disk layout (we use the `checkpoints/step_<N>/` form).
- `CLAUDE.md` "Normalization" section — checkpoints must preserve `action_mean`, `action_std`, etc.; we honor that by saving the entire `norm_stats` JSON payload verbatim.
- Plan 2 commit `d3045c9` — `data/norm_stats/libero_spatial.json` is the canonical payload format we serialize.
- Plan 3 commit `e0d392e` — `scripts/train.py` consumes resolved `cfg`; the saved meta uses the same shape so a future resume path can replay the run.

**Hard constraints from CLAUDE.md:**
- "Centralized normalization. ...checkpoint should not contain only model weights. It should also preserve full config, dataset version, action schema, normalization statistics, tokenizer / processor settings, model architecture settings."
- "Save all metadata required to reproduce training and deployment."
- Boundary: code lives under `src/vla_project/training/`, no model imports there beyond `nn.Module` typing.

---

## File Structure

**Create:**
- `src/vla_project/training/checkpoint.py`
- `tests/test_checkpoint.py`

**Modify:**
- `src/vla_project/training/__init__.py` (re-export the two functions)

**Do not modify:** `Trainer`, `scripts/train.py`, `models/`, `policies/`, `robots/`. Wiring into the trainer is out of scope (later plan).

---

## Task 1: `save_checkpoint` / `load_checkpoint` round-trip

**Files:**
- Create: `src/vla_project/training/checkpoint.py`
- Create: `tests/test_checkpoint.py`
- Modify: `src/vla_project/training/__init__.py`

- [ ] **Step 1: Write failing tests**

`tests/test_checkpoint.py`:

```python
"""Round-trip tests for save_checkpoint / load_checkpoint."""
import json
import subprocess
from pathlib import Path
from typing import Dict

import pytest
import torch
import torch.nn as nn

from vla_project.training.checkpoint import (
    load_checkpoint,
    save_checkpoint,
)


class _Toy(nn.Module):
    """Tiny module so checkpoint round-trip is fast and deterministic."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(3, 4)
        self.fc2 = nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(torch.relu(self.fc(x)))


def _state_dicts_equal(a: Dict[str, torch.Tensor], b: Dict[str, torch.Tensor]) -> bool:
    if set(a.keys()) != set(b.keys()):
        return False
    for k in a:
        if not torch.equal(a[k], b[k]):
            return False
    return True


def test_save_creates_files(tmp_path: Path) -> None:
    m = _Toy()
    cfg = {"train": {"lr": 1e-4, "batch_size": 1}, "model": {"num_blocks": 35}}
    out = tmp_path / "step_42"
    save_checkpoint(out, m, step=42, cfg=cfg)
    assert (out / "model.pt").is_file()
    assert (out / "meta.json").is_file()
    # No optimizer was passed, so optimizer.pt must not exist.
    assert not (out / "optimizer.pt").exists()


def test_load_restores_state_dict(tmp_path: Path) -> None:
    m1 = _Toy()
    m1.fc.weight.data.fill_(0.5)
    m1.fc2.weight.data.fill_(-0.25)
    out = tmp_path / "step_1"
    save_checkpoint(out, m1, step=1, cfg={"a": 1})

    m2 = _Toy()  # fresh random init
    assert not _state_dicts_equal(m1.state_dict(), m2.state_dict())
    meta = load_checkpoint(out, m2)
    assert _state_dicts_equal(m1.state_dict(), m2.state_dict())
    assert meta["step"] == 1
    assert meta["cfg"] == {"a": 1}


def test_save_records_norm_stats(tmp_path: Path) -> None:
    m = _Toy()
    norm_stats = {
        "libero_spatial_no_noops": {
            "action": {
                "q01": [-1.0] * 7,
                "q99": [ 1.0] * 7,
                "mask": [True, True, True, True, True, True, False],
            }
        }
    }
    out = tmp_path / "step_2"
    save_checkpoint(out, m, step=2, cfg={}, norm_stats=norm_stats)
    meta = json.loads((out / "meta.json").read_text())
    assert meta["norm_stats"] == norm_stats


def test_save_records_git_commit(tmp_path: Path) -> None:
    """git_commit should be a 40-char hex string OR 'unknown' if not in a repo."""
    m = _Toy()
    out = tmp_path / "step_3"
    save_checkpoint(out, m, step=3, cfg={})
    meta = json.loads((out / "meta.json").read_text())
    gc = meta["git_commit"]
    assert isinstance(gc, str)
    # Either 40-hex or 'unknown' or '<hex>-dirty'
    assert (len(gc) == 40 and all(c in "0123456789abcdef" for c in gc)) or gc == "unknown" or gc.endswith("-dirty")


def test_save_records_tokenizer_settings(tmp_path: Path) -> None:
    m = _Toy()
    out = tmp_path / "step_4"
    save_checkpoint(
        out, m, step=4, cfg={},
        tokenizer_settings={"model_name": "google/gemma-4-E2B", "max_len": 50},
    )
    meta = json.loads((out / "meta.json").read_text())
    assert meta["tokenizer_settings"] == {"model_name": "google/gemma-4-E2B", "max_len": 50}


def test_save_with_optimizer_round_trip(tmp_path: Path) -> None:
    m = _Toy()
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    # Take one step so the optimizer has non-trivial state (Adam moments).
    m(torch.randn(2, 3)).sum().backward()
    opt.step()
    opt.zero_grad()
    state_before = opt.state_dict()

    out = tmp_path / "step_5"
    save_checkpoint(out, m, step=5, cfg={}, optimizer=opt)
    assert (out / "optimizer.pt").is_file()

    m2 = _Toy()
    opt2 = torch.optim.AdamW(m2.parameters(), lr=1e-3)
    load_checkpoint(out, m2, optimizer=opt2)
    state_after = opt2.state_dict()

    # Compare optimizer state — Adam stores `exp_avg`, `exp_avg_sq`, `step`.
    assert state_before["state"].keys() == state_after["state"].keys()
    for pid in state_before["state"]:
        for k, v in state_before["state"][pid].items():
            v2 = state_after["state"][pid][k]
            if torch.is_tensor(v):
                assert torch.equal(v, v2), f"opt state mismatch on {pid}/{k}"
            else:
                assert v == v2


def test_load_missing_dir_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_checkpoint(tmp_path / "does_not_exist", _Toy())


def test_load_missing_model_pt_raises(tmp_path: Path) -> None:
    out = tmp_path / "step_6"
    out.mkdir()
    (out / "meta.json").write_text("{}")  # only meta.json
    with pytest.raises(FileNotFoundError):
        load_checkpoint(out, _Toy())


def test_save_atomic_rename(tmp_path: Path) -> None:
    """The final out_dir name must not exist until the write completes."""
    m = _Toy()
    out = tmp_path / "step_7"
    save_checkpoint(out, m, step=7, cfg={})
    # After save, the dir exists and looks normal.
    assert out.is_dir()
    # Sibling temp dir from atomic rename must not linger (we check there's no
    # ``step_7.tmp`` or similar leftover sibling).
    siblings = [p.name for p in tmp_path.iterdir()]
    assert siblings == ["step_7"]


def test_save_overwrites_existing(tmp_path: Path) -> None:
    """Re-saving to the same dir must replace the previous checkpoint atomically."""
    out = tmp_path / "step_8"
    save_checkpoint(out, _Toy(), step=8, cfg={"v": 1})
    save_checkpoint(out, _Toy(), step=8, cfg={"v": 2})
    meta = json.loads((out / "meta.json").read_text())
    assert meta["cfg"] == {"v": 2}


def test_resolved_omegaconf_serializes(tmp_path: Path) -> None:
    """OmegaConf DictConfig must be serializable via OmegaConf.to_container()."""
    from omegaconf import OmegaConf
    cfg = OmegaConf.create({
        "train": {"lr": 1e-4, "batch_size": 1},
        "data": {"type": "libero_synthetic"},
    })
    out = tmp_path / "step_9"
    save_checkpoint(out, _Toy(), step=9, cfg=cfg)
    meta = json.loads((out / "meta.json").read_text())
    assert meta["cfg"] == {
        "train": {"lr": 1e-4, "batch_size": 1},
        "data": {"type": "libero_synthetic"},
    }
```

- [ ] **Step 2: Run, expect FAIL**

```bash
PYTHONPATH="" uv run pytest tests/test_checkpoint.py -v
```

Expected: `ImportError: cannot import name 'save_checkpoint'`.

- [ ] **Step 3: Implement**

`src/vla_project/training/checkpoint.py`:

```python
"""Save / load self-contained checkpoint directories.

A checkpoint is a directory with the following layout::

  step_<N>/
  ├── model.pt        # torch.save(model.state_dict())
  ├── optimizer.pt    # torch.save(optimizer.state_dict())   [optional]
  └── meta.json       # step, cfg, norm_stats, git_commit, tokenizer_settings

Saves are atomic: the contents are first written to a sibling ``.tmp`` dir,
then renamed into place, so a crashed save cannot leave a half-baked dir.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn


def _git_commit() -> str:
    """Return the current HEAD commit, suffixed with ``-dirty`` if the working
    tree has uncommitted changes. Returns ``"unknown"`` if not in a repo."""
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"
    try:
        dirty = bool(subprocess.check_output(
            ["git", "status", "--porcelain"],
            stderr=subprocess.DEVNULL,
        ).decode().strip())
    except (subprocess.CalledProcessError, FileNotFoundError):
        dirty = False
    return f"{sha}-dirty" if dirty else sha


def _resolve_cfg(cfg: Any) -> Any:
    """Best-effort coerce OmegaConf DictConfig (or plain dict) into a plain
    JSON-serializable container, resolving ``${oc.env:...}`` interpolations."""
    try:
        from omegaconf import DictConfig, ListConfig, OmegaConf
    except ImportError:  # pragma: no cover — omegaconf is a project dep
        return cfg
    if isinstance(cfg, (DictConfig, ListConfig)):
        return OmegaConf.to_container(cfg, resolve=True)
    return cfg


def save_checkpoint(
    out_dir: Union[str, Path],
    model: nn.Module,
    *,
    step: int,
    cfg: Any,
    norm_stats: Optional[Dict[str, Any]] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    tokenizer_settings: Optional[Dict[str, Any]] = None,
) -> None:
    out = Path(out_dir)
    parent = out.parent
    parent.mkdir(parents=True, exist_ok=True)
    tmp = parent / (out.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir()

    torch.save(model.state_dict(), tmp / "model.pt")
    if optimizer is not None:
        torch.save(optimizer.state_dict(), tmp / "optimizer.pt")

    meta = {
        "step": int(step),
        "cfg": _resolve_cfg(cfg),
        "norm_stats": norm_stats,
        "tokenizer_settings": tokenizer_settings,
        "git_commit": _git_commit(),
    }
    (tmp / "meta.json").write_text(json.dumps(meta, indent=2))

    # Atomic replace: remove any existing dir at the target path, then rename.
    if out.exists():
        shutil.rmtree(out)
    tmp.rename(out)


def load_checkpoint(
    in_dir: Union[str, Path],
    model: nn.Module,
    *,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> Dict[str, Any]:
    in_path = Path(in_dir)
    if not in_path.is_dir():
        raise FileNotFoundError(f"checkpoint dir not found: {in_path}")
    model_pt = in_path / "model.pt"
    if not model_pt.is_file():
        raise FileNotFoundError(f"missing model.pt under {in_path}")
    meta_path = in_path / "meta.json"

    state = torch.load(model_pt, map_location="cpu", weights_only=True)
    model.load_state_dict(state)

    if optimizer is not None:
        opt_pt = in_path / "optimizer.pt"
        if not opt_pt.is_file():
            raise FileNotFoundError(f"missing optimizer.pt under {in_path}")
        optimizer.load_state_dict(torch.load(opt_pt, map_location="cpu", weights_only=False))

    if meta_path.is_file():
        return json.loads(meta_path.read_text())
    return {}
```

`src/vla_project/training/__init__.py` — append (or create with) the re-exports:

```python
from vla_project.training.checkpoint import load_checkpoint, save_checkpoint  # noqa: F401
```

(If `__init__.py` already has other re-exports, append these and preserve existing lines. Do not delete or reorder.)

- [ ] **Step 4: Run, expect PASS**

```bash
PYTHONPATH="" uv run pytest tests/test_checkpoint.py -v
PYTHONPATH="" uv run pytest -q
```

Expected: 10 new tests pass; full suite green (current 63 + 10 = 73 expected).

- [ ] **Step 5: Commit**

```bash
git add src/vla_project/training/checkpoint.py \
        src/vla_project/training/__init__.py \
        tests/test_checkpoint.py
git commit -m "feat(training): save/load checkpoint dirs with cfg + norm_stats + git"
```

---

## Task 2: VLAPolicy round-trip integration test

**Files:**
- Create: `tests/test_checkpoint_vla_policy.py`

This task verifies the checkpoint round-trip on the actual `VLAPolicy` (using stub vision/language modules per existing test fixtures), proving the contract holds for our real `state_dict()` shape — not just toy modules.

- [ ] **Step 1: Inspect existing stub fixtures**

```bash
grep -nE "_StubGemma|_StubSig" tests/_stubs.py 2>&1 | head -10
```

Expected: the stubs exist (added in earlier plans). Note their constructor signatures — the integration test should reuse them as-is.

- [ ] **Step 2: Write the integration test**

`tests/test_checkpoint_vla_policy.py`:

```python
"""Integration round-trip: save + load on the real VLAPolicy state_dict."""
from pathlib import Path

import torch

from vla_project.data import constants as C
from vla_project.models.vla_policy import VLAPolicy, VLAPolicyConfig
from vla_project.training.checkpoint import load_checkpoint, save_checkpoint
from tests._stubs import _StubGemma, _StubSig


def _make_policy() -> VLAPolicy:
    cfg = VLAPolicyConfig(num_domains=1, num_blocks=2, num_action_queries=4, num_soft_prompt_tokens=4)
    return VLAPolicy(cfg, _StubSig(), _StubGemma())


def test_vla_policy_round_trip(tmp_path: Path) -> None:
    p1 = _make_policy()
    # Mutate one parameter so the saved state diverges from a fresh init.
    with torch.no_grad():
        p1.action_decoder.weight.fill_(0.123)

    out = tmp_path / "step_100"
    save_checkpoint(out, p1, step=100, cfg={"smoke": True})

    p2 = _make_policy()
    meta = load_checkpoint(out, p2)

    sd1 = p1.state_dict()
    sd2 = p2.state_dict()
    assert set(sd1.keys()) == set(sd2.keys())
    for k in sd1:
        assert torch.equal(sd1[k], sd2[k]), f"mismatch at {k}"
    assert meta["step"] == 100
```

If `VLAPolicyConfig`'s defaults differ from what tests/_stubs.py supports (e.g., the stub Gemma has fixed num_layers), adjust `num_blocks` accordingly. The goal is shape-correctness and round-trip equality, not behavioral correctness.

- [ ] **Step 3: Run**

```bash
PYTHONPATH="" uv run pytest tests/test_checkpoint_vla_policy.py -v
```

Expected: 1 passed. If a constructor mismatch breaks the test, examine `tests/_stubs.py` and adapt the integration test rather than the stubs (the stubs are shared across many tests).

- [ ] **Step 4: Full pytest**

```bash
PYTHONPATH="" uv run pytest -q
```

Expected: 74 green (10 from Task 1 + 1 from Task 2 + 63 prior).

- [ ] **Step 5: Commit**

```bash
git add tests/test_checkpoint_vla_policy.py
git commit -m "test(training): VLAPolicy state_dict round-trip via save/load_checkpoint"
```

---

## Task 3: Push branch + open PR

- [ ] **Step 1: Confirm branch**

```bash
git status -sb
git log --oneline feat/multi-domain-sampler..HEAD
```

The controller should already have created `feat/checkpoint-save-load` branched from `feat/multi-domain-sampler` before dispatching Task 1.

- [ ] **Step 2: Push**

```bash
git push -u origin feat/checkpoint-save-load
```

- [ ] **Step 3: PR**

PR base: `feat/multi-domain-sampler` (rebase to `main` once Plans 1-3 merge).
Title: `feat(training): self-contained checkpoint dir save/load`.
Body should include the test count delta and a note that wiring into `Trainer` / `scripts/train.py` is intentionally deferred.

---

## Done criteria

- [ ] `uv run pytest -q` passes (full suite, 11 new tests).
- [ ] Round-trip on `_Toy` and on `VLAPolicy` both restore bit-exact state_dicts.
- [ ] `meta.json` includes `step`, `cfg`, `norm_stats`, `git_commit`, `tokenizer_settings` fields.
- [ ] `Trainer` and `scripts/train.py` unchanged.
- [ ] No edits to `models/`, `policies/`, `robots/`.

## Out of scope (later plans)

- Wiring `save_checkpoint(...)` calls into `Trainer.fit` (auto-save every N steps).
- A `--resume <path>` CLI flag in `scripts/train.py`.
- Pushing checkpoints to HF Hub or any remote storage.
- Compressing or deduplicating large weight tensors (we keep `torch.save` defaults).
