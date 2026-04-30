# compute_norm_stats Tool + Persisted Stats Integration (Plan 2 / 10)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stop relying on the borrowed host-specific stats JSON at `/misc/dl00/takaki/vla-gemma-4/...`. Add a CLI tool that computes BOUNDS_Q99 stats from a LeRobot LIBERO dataset, persist the produced JSON inside the repo at `data/norm_stats/libero_spatial.json`, and switch `configs/data/libero_real.yaml` to point at the in-repo file. Verify the real-data smoke remains green with the freshly computed stats.

**Architecture:** Add a pure compute function `compute_q99_stats(action_arr, mask) -> Q99Stats` to the existing `data/normalization.py` (input: `[N, A]` numpy/tensor, output: `Q99Stats`). The CLI in `tools/compute_norm_stats.py` instantiates `LeRobotDataset` (no `delta_timestamps` so it yields per-frame actions), accumulates raw actions, calls the pure compute, and writes a project-schema JSON file. The JSON is committed to the repo so smoke runs are portable across hosts. The data layer (`src/vla_project/data/`) remains lerobot-free — only the tool imports lerobot.

**Tech Stack:** numpy (already a dep), the existing `lerobot.datasets.LeRobotDataset` (added in Plan 1), `argparse` for CLI.

**Repo references:**
- `src/vla_project/data/normalization.py` — the canonical normalization module; we extend it.
- `/misc/dl00/takaki/vla-gemma-4/VLA-Adapter/outputs/LIBERO-Spatial-Pro/dataset_statistics.json` — the legacy borrowed stats; we mirror its schema (`{"<dataset_key>": {"action": {"q01": [...], "q99": [...], "mask": [...], "mean": [...], "std": [...], "min": [...], "max": [...]}}}`).
- `docs/architectures/x_vla_adapter.md` — confirms the locked schema for action stats (q01 / q99 / mask).

**Hard constraints from CLAUDE.md:**
- Centralized normalization in `data/normalization.py`. The compute function lives there.
- `tools/` may be less polished than `src/`, but must not duplicate core normalization logic — it must call `compute_q99_stats`, not re-implement.
- Configs drive paths. `configs/data/libero_real.yaml` switches to the in-repo JSON.
- No hardcoded paths in `src/`. CLI in `tools/` may take args; relative defaults are fine.
- Smoke compatibility: the real-data smoke train must still pass with the freshly computed stats.

---

## File Structure

**Create:**
- `tools/__init__.py` (empty file so the dir is importable for tests; tools dir does not yet exist)
- `tools/compute_norm_stats.py`
- `data/norm_stats/.gitkeep`
- `data/norm_stats/libero_spatial.json` (output of the tool, committed for portability)
- `tests/test_compute_q99_stats.py`

**Modify:**
- `src/vla_project/data/normalization.py` (append `compute_q99_stats(...)`; do not touch existing symbols)
- `configs/data/libero_real.yaml` (point `stats_path` at the new in-repo JSON via env-var fallback so dl40 default still works)
- `configs/train/smoke_real.yaml` (mirror the same `stats_path` change)
- `.gitignore` (verify `data/norm_stats/` is NOT gitignored — fix if needed)

---

## Task 1: `compute_q99_stats` pure function

**Files:**
- Modify: `src/vla_project/data/normalization.py` (append only — do NOT edit existing `Q99Stats`, `load_q99_stats`, `normalize_action_q99`)
- Create: `tests/test_compute_q99_stats.py`

- [ ] **Step 1: Write failing tests**

`tests/test_compute_q99_stats.py`:

```python
"""Tests for the pure `compute_q99_stats` helper added to data/normalization.py."""
from typing import List

import numpy as np
import pytest
import torch

from vla_project.data.normalization import (
    Q99Stats,
    compute_q99_stats,
    normalize_action_q99,
)


def _ramp_actions(n: int = 1000, a: int = 7) -> np.ndarray:
    """Linear ramp from -10 to 10 on each dim, shifted per dim, with a binary final dim."""
    base = np.linspace(-10.0, 10.0, n, dtype=np.float32)  # [N]
    arr = np.stack([base + i for i in range(a - 1)], axis=1)  # [N, A-1]
    gripper = (np.arange(n) % 2).astype(np.float32).reshape(-1, 1)  # binary 0/1
    return np.concatenate([arr, gripper], axis=1)  # [N, A]


def test_returns_q99_stats_with_correct_shapes() -> None:
    arr = _ramp_actions(500, 7)
    mask = [True, True, True, True, True, True, False]
    stats = compute_q99_stats(arr, mask=mask)
    assert isinstance(stats, Q99Stats)
    assert stats.q01.shape == (7,)
    assert stats.q99.shape == (7,)
    assert stats.mask.shape == (7,)
    assert stats.mask.dtype == torch.bool
    assert stats.mask.tolist() == mask


def test_default_mask_is_all_true() -> None:
    arr = _ramp_actions(100, 7)
    stats = compute_q99_stats(arr)
    assert stats.mask.tolist() == [True] * 7


def test_q01_q99_match_numpy_quantiles() -> None:
    arr = _ramp_actions(10000, 7)
    stats = compute_q99_stats(arr)
    expected_q01 = np.quantile(arr, 0.01, axis=0)
    expected_q99 = np.quantile(arr, 0.99, axis=0)
    assert np.allclose(stats.q01.numpy(), expected_q01, atol=1e-4)
    assert np.allclose(stats.q99.numpy(), expected_q99, atol=1e-4)


def test_round_trip_into_normalize() -> None:
    """Computed stats applied via normalize_action_q99 produce values in [-1, 1] on mask=True dims."""
    arr = _ramp_actions(1000, 7)
    mask = [True] * 6 + [False]
    stats = compute_q99_stats(arr, mask=mask)
    normed = normalize_action_q99(torch.from_numpy(arr), stats)
    # mask=True dims fall in [-1, 1] (within numerical slack)
    assert normed[:, :6].abs().max().item() <= 1.0 + 1e-6
    # mask=False dim passes through (still 0 or 1)
    assert torch.all((normed[:, 6] == 0.0) | (normed[:, 6] == 1.0))


def test_accepts_torch_input() -> None:
    arr = torch.from_numpy(_ramp_actions(200, 7))
    stats = compute_q99_stats(arr)
    assert stats.q01.shape == (7,)


def test_rejects_wrong_rank() -> None:
    arr = np.zeros((10, 8, 7), dtype=np.float32)  # 3-D input
    with pytest.raises(ValueError):
        compute_q99_stats(arr)


def test_rejects_mask_length_mismatch() -> None:
    arr = _ramp_actions(50, 7)
    with pytest.raises(ValueError):
        compute_q99_stats(arr, mask=[True, True, True])  # only 3 entries
```

- [ ] **Step 2: Run, expect FAIL**

```bash
PYTHONPATH="" uv run pytest tests/test_compute_q99_stats.py -v
```

Expected: `ImportError: cannot import name 'compute_q99_stats'`.

- [ ] **Step 3: Implement (append to `normalization.py`)**

Append to `src/vla_project/data/normalization.py`:

```python
import numpy as np


def compute_q99_stats(
    action_arr: "np.ndarray | torch.Tensor",
    mask: "list[bool] | None" = None,
) -> Q99Stats:
    """Compute BOUNDS_Q99 stats from a 2-D array of raw actions.

    Args:
        action_arr: shape ``[N, A]`` (numpy ndarray or torch.Tensor). Each row
            is one raw action vector. Larger N gives more stable quantiles.
        mask: list of ``A`` bools indicating which dims to normalize (``True``)
            vs pass through (``False``). Defaults to all-True. Length must match
            ``action_arr.shape[-1]``.

    Returns:
        Q99Stats with ``q01``, ``q99``, ``mask`` as float32 / bool tensors of
        shape ``[A]``.
    """
    if torch.is_tensor(action_arr):
        arr = action_arr.detach().cpu().numpy()
    else:
        arr = np.asarray(action_arr)
    if arr.ndim != 2:
        raise ValueError(
            f"action_arr must be 2-D [N, A]; got rank {arr.ndim} shape {arr.shape}"
        )
    A = arr.shape[1]
    if mask is None:
        mask = [True] * A
    if len(mask) != A:
        raise ValueError(
            f"mask length {len(mask)} != action dim {A}"
        )
    q01 = np.quantile(arr.astype(np.float32), 0.01, axis=0).astype(np.float32)
    q99 = np.quantile(arr.astype(np.float32), 0.99, axis=0).astype(np.float32)
    return Q99Stats(
        q01=torch.from_numpy(q01),
        q99=torch.from_numpy(q99),
        mask=torch.tensor(mask, dtype=torch.bool),
    )
```

(`import numpy as np` goes at the top of the file alongside `import torch` if not already there. If you find a top-level `import numpy as np` already exists, do not duplicate it.)

- [ ] **Step 4: Run, expect PASS**

```bash
PYTHONPATH="" uv run pytest tests/test_compute_q99_stats.py -v
PYTHONPATH="" uv run pytest -q   # full suite still green
```

- [ ] **Step 5: Commit**

```bash
git add src/vla_project/data/normalization.py tests/test_compute_q99_stats.py
git commit -m "feat(data): compute_q99_stats helper + roundtrip test"
```

---

## Task 2: CLI `tools/compute_norm_stats.py`

**Files:**
- Create: `tools/__init__.py`
- Create: `tools/compute_norm_stats.py`

- [ ] **Step 1: Add empty `tools/__init__.py`**

```bash
mkdir -p tools
touch tools/__init__.py
```

- [ ] **Step 2: Implement the CLI**

`tools/compute_norm_stats.py`:

```python
"""Compute BOUNDS_Q99 action stats from a LeRobot LIBERO dataset and write JSON.

The output JSON matches the project schema consumed by
`vla_project.data.normalization.load_q99_stats`:

  {
    "<dataset_key>": {
      "action": {
        "q01":  [..., A floats],
        "q99":  [..., A floats],
        "mask": [..., A bools],
        "mean": [..., A floats],
        "std":  [..., A floats],
        "min":  [..., A floats],
        "max":  [..., A floats]
      }
    }
  }

The mean / std / min / max blocks are recorded for forward compatibility (the
legacy dataset_statistics.json includes them) but the project consumes only
q01 / q99 / mask via `load_q99_stats`.

Usage (from repo root):

  PYTHONPATH="" uv run python tools/compute_norm_stats.py \\
    --repo_id lerobot/libero_spatial_image \\
    --dataset_key libero_spatial_no_noops \\
    --output data/norm_stats/libero_spatial.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from tqdm import tqdm

from vla_project.data.normalization import compute_q99_stats


# LIBERO single-arm Franka: 6 EE delta dims + 1 gripper binary; gripper = mask=False.
_LIBERO_DEFAULT_MASK: List[bool] = [True] * 6 + [False]


def _collect_actions(repo_id: str, episodes: Optional[List[int]]) -> np.ndarray:
    """Iterate the LeRobot dataset and stack per-frame actions to [N, A]."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    # No `delta_timestamps`: one action per frame, shape [A].
    ds = LeRobotDataset(repo_id, episodes=episodes, download_videos=False)

    rows: List[np.ndarray] = []
    for i in tqdm(range(len(ds)), desc=f"reading {repo_id}"):
        sample = ds[i]
        a = sample["action"]
        if torch.is_tensor(a):
            a = a.detach().cpu().numpy()
        rows.append(np.asarray(a, dtype=np.float32))
    if not rows:
        raise RuntimeError(f"no frames found in {repo_id} (episodes={episodes!r})")
    return np.stack(rows, axis=0)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--repo_id", required=True, help="LeRobot HF dataset repo id")
    p.add_argument("--dataset_key", required=True, help="Key under which to store the action block in the output JSON")
    p.add_argument("--output", required=True, type=Path, help="Path to write the JSON")
    p.add_argument("--episodes", type=int, nargs="*", default=None, help="Subset of episode indices (default: all)")
    p.add_argument("--mask", type=int, nargs="*", default=None,
                   help="Per-dim mask as 0/1 ints (default: LIBERO single-arm "
                        "[1,1,1,1,1,1,0])")
    args = p.parse_args()

    arr = _collect_actions(args.repo_id, args.episodes)
    mask: List[bool]
    if args.mask is None:
        mask = list(_LIBERO_DEFAULT_MASK)
    else:
        mask = [bool(x) for x in args.mask]
    if arr.shape[1] != len(mask):
        raise ValueError(
            f"action dim {arr.shape[1]} != mask length {len(mask)}; pass --mask"
        )

    stats = compute_q99_stats(arr, mask=mask)
    payload = {
        args.dataset_key: {
            "action": {
                "q01":  stats.q01.tolist(),
                "q99":  stats.q99.tolist(),
                "mask": [bool(b) for b in stats.mask.tolist()],
                "mean": np.mean(arr, axis=0).astype(float).tolist(),
                "std":  np.std(arr, axis=0).astype(float).tolist(),
                "min":  np.min(arr, axis=0).astype(float).tolist(),
                "max":  np.max(arr, axis=0).astype(float).tolist(),
            }
        }
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2))
    print(f"[compute_norm_stats] wrote {args.output} (N={arr.shape[0]}, A={arr.shape[1]})")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Smoke-import the CLI module (no run yet)**

```bash
PYTHONPATH="" uv run python -c "import tools.compute_norm_stats as m; print('ok', m.__doc__.splitlines()[0])"
```

Expected: `ok Compute BOUNDS_Q99 action stats from a LeRobot LIBERO dataset and write JSON.`

(No commit yet — the next task runs the CLI and commits the output JSON together with this script.)

---

## Task 3: Run CLI on `lerobot/libero_spatial_image` and commit JSON

**Files:**
- Create: `data/norm_stats/.gitkeep`
- Create: `data/norm_stats/libero_spatial.json` (CLI output)

- [ ] **Step 1: Confirm `data/norm_stats/` is NOT gitignored**

```bash
grep -nE 'data/|^data$|^data\b' .gitignore || echo "data/ not in gitignore — safe"
```

If a line ignores `data/` wholesale, modify `.gitignore` to allow `data/norm_stats/` (e.g., `!data/norm_stats/`). Commit the `.gitignore` edit alongside the JSON in this task.

- [ ] **Step 2: Run the CLI to produce the JSON**

```bash
mkdir -p data/norm_stats
PYTHONPATH="" uv run python tools/compute_norm_stats.py \
    --repo_id lerobot/libero_spatial_image \
    --dataset_key libero_spatial_no_noops \
    --output data/norm_stats/libero_spatial.json
```

Expected: prints `[compute_norm_stats] wrote data/norm_stats/libero_spatial.json (N=<n>, A=7)` after a few minutes (one full pass over 432 episodes / ~52k frames). The JSON file is created.

- [ ] **Step 3: Sanity-check the JSON shape**

```bash
PYTHONPATH="" uv run python -c "
import json
p = 'data/norm_stats/libero_spatial.json'
d = json.load(open(p))
a = d['libero_spatial_no_noops']['action']
print('keys:', list(a.keys()))
print('A:', len(a['q01']))
print('q01:', a['q01'])
print('q99:', a['q99'])
print('mask:', a['mask'])
"
```

Expected output: `keys` includes q01/q99/mask/mean/std/min/max; `A` is 7; `mask` is `[True, True, True, True, True, True, False]`; numerical q01/q99 are not all zero.

- [ ] **Step 4: Verify the new JSON loads via `load_q99_stats`**

```bash
PYTHONPATH="" uv run python -c "
from vla_project.data.normalization import load_q99_stats
s = load_q99_stats('data/norm_stats/libero_spatial.json', 'libero_spatial_no_noops')
print('q01:', s.q01.tolist())
print('q99:', s.q99.tolist())
print('mask:', s.mask.tolist())
"
```

Expected: same values as Step 3, all tensors of length 7.

- [ ] **Step 5: Add `.gitkeep` and commit**

```bash
touch data/norm_stats/.gitkeep
git add tools/__init__.py tools/compute_norm_stats.py \
        data/norm_stats/.gitkeep data/norm_stats/libero_spatial.json
# also include .gitignore if you edited it in Step 1
git diff --cached --name-only   # confirm exactly the files above
git commit -m "feat(tools): compute_norm_stats CLI + LIBERO-Spatial stats JSON"
```

---

## Task 4: Switch `configs/data/libero_real.yaml` to the in-repo JSON

**Files:**
- Modify: `configs/data/libero_real.yaml`
- Modify: `configs/train/smoke_real.yaml`

- [ ] **Step 1: Edit `configs/data/libero_real.yaml`**

Replace the `# TODO(portability): ...` block + the absolute `stats_path` line with:

```yaml
# stats_path resolves to the in-repo computed JSON by default; override with
# the `LIBERO_STATS_PATH` env var if running against a custom dataset.
stats_path: ${oc.env:LIBERO_STATS_PATH,data/norm_stats/libero_spatial.json}
```

OmegaConf's `${oc.env:VAR,default}` resolves at load time. `data/norm_stats/libero_spatial.json` is a path relative to the cwd of the script; since `scripts/train.py` is invoked from the repo root, this works.

- [ ] **Step 2: Mirror the change in `configs/train/smoke_real.yaml`**

The duplicated `data:` block in `smoke_real.yaml` keeps its own `stats_path:` line; update it identically:

```yaml
  stats_path: ${oc.env:LIBERO_STATS_PATH,data/norm_stats/libero_spatial.json}
```

- [ ] **Step 3: Commit**

```bash
git add configs/data/libero_real.yaml configs/train/smoke_real.yaml
git commit -m "feat(configs): use in-repo norm-stats JSON with env-var override"
```

---

## Task 5: Real-data smoke verification with new stats

**Files:** none (run-only).

- [ ] **Step 1: Run real-data smoke**

```bash
cd /misc/dl00/takaki/X-VLA-Adapter
CUDA_VISIBLE_DEVICES=3 PYTHONPATH="" timeout 600 \
    uv run python scripts/train.py configs/train/smoke_real.yaml 2>&1 | tee /tmp/smoke_real_v2.log
```

Expected: `[train] losses=[<f1>, <f2>]` with two finite floats. The numerical values may differ from Plan 1's run (`losses=[0.696, 2.690]`) because the new q01/q99 are computed from the full dataset rather than borrowed from another project — a small drift is expected and acceptable as long as both losses are finite.

- [ ] **Step 2: Run the override path**

```bash
LIBERO_STATS_PATH=/misc/dl00/takaki/vla-gemma-4/VLA-Adapter/outputs/LIBERO-Spatial-Pro/dataset_statistics.json \
CUDA_VISIBLE_DEVICES=3 PYTHONPATH="" timeout 600 \
    uv run python scripts/train.py configs/train/smoke_real.yaml 2>&1 | tail -3
```

Expected: still two finite floats. This proves the env-var override resolves correctly.

- [ ] **Step 3: Full pytest**

```bash
PYTHONPATH="" uv run pytest -q
```

Expected: all green (47 from before + 7 new compute_q99_stats tests = 54 or thereabouts).

No commit in this task — verification only.

---

## Task 6: Push branch + open PR

**Files:** none changed.

- [ ] **Step 1: Branch from `feat/real-libero-loader` if not already**

```bash
git status -sb   # confirm we're on a Plan-2 branch already, OR
git checkout -b feat/compute-norm-stats feat/real-libero-loader
```

(The controller may have already created `feat/compute-norm-stats` before dispatching Task 1. Confirm before branching.)

- [ ] **Step 2: Push**

```bash
git push -u origin feat/compute-norm-stats
```

- [ ] **Step 3: Note PR URL for the user**

GitHub will print a `Create a pull request` URL. Surface it to the user. PR base should be `main` *unless* `feat/real-libero-loader` (Plan 1) has not yet merged — in that case set base to `feat/real-libero-loader` and rebase onto `main` once Plan 1 merges.

PR title: `feat(tools): compute_norm_stats CLI + persisted LIBERO stats`

PR body includes:
- Plan reference: `docs/superpowers/plans/2026-04-30-compute-norm-stats.md`
- Smoke evidence: paste the two `[train] losses=[...]` lines from Steps 1 & 2 of Task 5.
- Test count delta: before vs after.
- Note that the borrowed `/misc/dl00/takaki/vla-gemma-4/...` path is no longer required; the env-var override keeps it as an escape hatch.

---

## Done criteria

- [ ] `uv run pytest -q` passes (full suite, including 7 new `compute_q99_stats` tests).
- [ ] `data/norm_stats/libero_spatial.json` exists in the repo and loads via `load_q99_stats`.
- [ ] `python scripts/train.py configs/train/smoke_real.yaml` runs ≥ 2 steps without NaN, using only in-repo files.
- [ ] `LIBERO_STATS_PATH=<other path> python scripts/train.py configs/train/smoke_real.yaml` honors the override.
- [ ] No edits to `models/`, `policies/`, `robots/`, `training/`.
- [ ] No new lerobot imports under `src/vla_project/data/` (the import remains only in `tools/compute_norm_stats.py`).

## Out of scope (other plans)

- Multi-domain stats (Plan 3 will compute per-domain `dataset_key` blocks for goal/object/10).
- Persisting norm_stats inside checkpoints (Plan 4 will bundle them into the checkpoint metadata).
