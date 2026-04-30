# Multi-domain Weighted Sampler Implementation Plan (Plan 3 / 10)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Plumb the data-side of multi-domain training. Add a `WeightedMultiDataset` mixer that draws samples from multiple per-domain `LeRobotLiberoDataset` instances with configurable mixing weights, set each sample's `domain_id` from the source domain, and verify a smoke train passes with `num_domains > 1` on real LIBERO splits (spatial + goal). The action head's `SoftPromptHub` and `DomainAwareLinear` already accept arbitrary `domain_id`; this plan only changes data ingest.

**Architecture:** New `data/datasets/weighted_multi_dataset.py` — `IterableDataset` that holds a list of child `LeRobotLiberoDataset` instances, normalizes user-supplied weights, and on each draw picks a child via `np.random.choice` then yields its next sample. Children are restarted when their iterator exhausts so the mixer is effectively infinite. `train.py` gains a new dispatch branch `cfg.data.type == "libero_lerobot_multidomain"` that constructs N children from a YAML list. To exercise the path end-to-end, compute BOUNDS_Q99 stats for `lerobot/libero_goal_image` and run a 2-domain smoke (spatial + goal, weights = [1, 1]) with `num_domains: 2`.

**Tech Stack:** existing `LeRobotLiberoDataset` (Plan 1), existing `compute_norm_stats` CLI (Plan 2), `numpy.random.Generator`, OmegaConf list configs.

**Repo references:**
- `src/vla_project/data/datasets/lerobot_libero_dataset.py` — child dataset class. Re-used as-is.
- `src/vla_project/models/projectors/domain_aware_linear.py`, `soft_prompts.py` — already accept any `domain_id ∈ [0, num_domains)`.
- `tests/test_domain_aware_swap_full.py` — confirms different `domain_id` produces different projections; this is the architectural prereq for Plan 3.
- `tools/compute_norm_stats.py` — re-used for the goal split.
- `docs/architectures/x_vla_adapter.md` — confirms `domain_id` is a per-sample integer in `[0, num_domains)`.

**Hard constraints from CLAUDE.md:**
- New code stays in `src/vla_project/data/datasets/` and `tools/` only. Models / policies / training core untouched.
- The mixer fails fast: weights non-empty, weights non-negative, weights sum > 0, child datasets non-empty.
- Configs drive everything (no hardcoded HF repo IDs in src/).
- Smoke test is the binary success criterion: forward + backward on the 2-domain real batch with no NaN.

---

## File Structure

**Create:**
- `src/vla_project/data/datasets/weighted_multi_dataset.py`
- `tests/test_weighted_multi_dataset.py`
- `data/norm_stats/libero_goal.json` (CLI output)
- `configs/data/libero_multidomain.yaml`
- `configs/train/smoke_multidomain.yaml`

**Modify:**
- `src/vla_project/data/datasets/__init__.py` (re-export `WeightedMultiDataset`)
- `scripts/train.py` (add `libero_lerobot_multidomain` dispatch branch)

**Do not modify:** `src/vla_project/models/`, `src/vla_project/policies/`, existing dataset class, existing tests.

---

## Task 1: `WeightedMultiDataset` — pure mixer

**Files:**
- Create: `src/vla_project/data/datasets/weighted_multi_dataset.py`
- Create: `tests/test_weighted_multi_dataset.py`
- Modify: `src/vla_project/data/datasets/__init__.py`

- [ ] **Step 1: Write failing tests**

`tests/test_weighted_multi_dataset.py`:

```python
"""Stochastic tests for WeightedMultiDataset.

Uses two trivial in-memory IterableDatasets that yield deterministic per-sample
domain_ids so we can statistically verify the mixer's draw distribution.
"""
from typing import Iterator, List

import numpy as np
import pytest
import torch
from torch.utils.data import IterableDataset

from vla_project.data.datasets.weighted_multi_dataset import WeightedMultiDataset


class _ConstDomainDataset(IterableDataset):
    """Yields {"domain_id": <fixed int>, "value": rand} forever."""

    def __init__(self, domain_id: int) -> None:
        super().__init__()
        self.domain_id = domain_id

    def __iter__(self) -> Iterator[dict]:
        while True:
            yield {
                "domain_id": torch.tensor(self.domain_id, dtype=torch.long),
                "value": torch.randn(1),
            }


def _draw_domain_ids(mix: WeightedMultiDataset, n: int) -> List[int]:
    out: List[int] = []
    it = iter(mix)
    for _ in range(n):
        s = next(it)
        out.append(int(s["domain_id"].item()))
    return out


def test_weights_one_zero_returns_only_first() -> None:
    mix = WeightedMultiDataset(
        datasets=[_ConstDomainDataset(0), _ConstDomainDataset(1)],
        weights=[1.0, 0.0],
        seed=0,
    )
    ids = _draw_domain_ids(mix, n=200)
    assert set(ids) == {0}


def test_equal_weights_split_roughly_half() -> None:
    mix = WeightedMultiDataset(
        datasets=[_ConstDomainDataset(0), _ConstDomainDataset(1)],
        weights=[1.0, 1.0],
        seed=0,
    )
    ids = _draw_domain_ids(mix, n=2000)
    frac = sum(1 for x in ids if x == 0) / len(ids)
    # Allow generous tolerance — this is a smoke test, not a chi-squared.
    assert 0.45 <= frac <= 0.55


def test_three_to_one_weights_match() -> None:
    mix = WeightedMultiDataset(
        datasets=[_ConstDomainDataset(0), _ConstDomainDataset(1)],
        weights=[3.0, 1.0],
        seed=0,
    )
    ids = _draw_domain_ids(mix, n=4000)
    frac0 = sum(1 for x in ids if x == 0) / len(ids)
    assert 0.70 <= frac0 <= 0.80


def test_seed_reproducible() -> None:
    a = _draw_domain_ids(
        WeightedMultiDataset(
            datasets=[_ConstDomainDataset(0), _ConstDomainDataset(1)],
            weights=[1.0, 1.0],
            seed=42,
        ),
        n=100,
    )
    b = _draw_domain_ids(
        WeightedMultiDataset(
            datasets=[_ConstDomainDataset(0), _ConstDomainDataset(1)],
            weights=[1.0, 1.0],
            seed=42,
        ),
        n=100,
    )
    assert a == b


def test_rejects_empty_datasets() -> None:
    with pytest.raises(ValueError):
        WeightedMultiDataset(datasets=[], weights=[])


def test_rejects_weight_length_mismatch() -> None:
    with pytest.raises(ValueError):
        WeightedMultiDataset(
            datasets=[_ConstDomainDataset(0), _ConstDomainDataset(1)],
            weights=[1.0],
        )


def test_rejects_negative_weights() -> None:
    with pytest.raises(ValueError):
        WeightedMultiDataset(
            datasets=[_ConstDomainDataset(0), _ConstDomainDataset(1)],
            weights=[1.0, -1.0],
        )


def test_rejects_zero_total_weight() -> None:
    with pytest.raises(ValueError):
        WeightedMultiDataset(
            datasets=[_ConstDomainDataset(0), _ConstDomainDataset(1)],
            weights=[0.0, 0.0],
        )
```

- [ ] **Step 2: Run, expect FAIL**

```bash
PYTHONPATH="" uv run pytest tests/test_weighted_multi_dataset.py -v
```

Expected: `ModuleNotFoundError` on the new module.

- [ ] **Step 3: Implement**

`src/vla_project/data/datasets/weighted_multi_dataset.py`:

```python
"""Weighted infinite mixer over multiple IterableDatasets.

Each child yields per-sample dicts (assumed to already include a ``domain_id``
field; the mixer does not inject one). On each draw, the mixer picks a child
index ``i`` from a categorical distribution proportional to the supplied
``weights`` and yields the next sample from child ``i``. When a child's
iterator exhausts, the mixer restarts that child's iterator (so the mix is
effectively infinite even if the children are finite).

This is the data-side analogue of X-VLA's `DATA_WEIGHTS` weighted sampler.
"""
from __future__ import annotations

from typing import Iterator, List, Optional, Sequence

import numpy as np
from torch.utils.data import IterableDataset


class WeightedMultiDataset(IterableDataset):
    def __init__(
        self,
        datasets: Sequence[IterableDataset],
        weights: Sequence[float],
        seed: Optional[int] = None,
    ) -> None:
        super().__init__()
        if len(datasets) == 0:
            raise ValueError("datasets is empty")
        if len(datasets) != len(weights):
            raise ValueError(
                f"len(weights)={len(weights)} != len(datasets)={len(datasets)}"
            )
        w = np.asarray(weights, dtype=np.float64)
        if (w < 0).any():
            raise ValueError(f"negative weight in {list(weights)!r}")
        total = float(w.sum())
        if total <= 0.0:
            raise ValueError(f"weights sum to {total!r}; must be > 0")
        self._datasets: List[IterableDataset] = list(datasets)
        self._probs: np.ndarray = w / total
        self._seed = seed

    def __iter__(self) -> Iterator[dict]:
        rng = np.random.default_rng(self._seed)
        iters: List[Iterator] = [iter(d) for d in self._datasets]
        n = len(self._datasets)
        while True:
            idx = int(rng.choice(n, p=self._probs))
            try:
                yield next(iters[idx])
            except StopIteration:
                iters[idx] = iter(self._datasets[idx])
                yield next(iters[idx])
```

`src/vla_project/data/datasets/__init__.py` — append:

```python
from vla_project.data.datasets.weighted_multi_dataset import WeightedMultiDataset  # noqa: F401
```

- [ ] **Step 4: Run, expect PASS**

```bash
PYTHONPATH="" uv run pytest tests/test_weighted_multi_dataset.py -v
PYTHONPATH="" uv run pytest -q
```

Expected: 8/8 new tests pass; full suite stays green.

- [ ] **Step 5: Commit**

```bash
git add src/vla_project/data/datasets/weighted_multi_dataset.py \
        src/vla_project/data/datasets/__init__.py \
        tests/test_weighted_multi_dataset.py
git commit -m "feat(data): WeightedMultiDataset mixer over IterableDatasets"
```

---

## Task 2: Compute LIBERO-Goal stats

**Files:**
- Create: `data/norm_stats/libero_goal.json`

This task re-uses the Plan-2 CLI to produce a second per-domain stats file. No code change.

- [ ] **Step 1: Run the CLI**

```bash
cd /misc/dl00/takaki/X-VLA-Adapter
PYTHONPATH="" uv run python tools/compute_norm_stats.py \
    --repo_id lerobot/libero_goal_image \
    --dataset_key libero_goal_no_noops \
    --output data/norm_stats/libero_goal.json
```

Expected: prints `[compute_norm_stats] wrote data/norm_stats/libero_goal.json (N=<n>, A=7)` after parquet snapshot download (~1-3 min cold) + read (~2 s).

If `lerobot/libero_goal_image` is not on the HF hub, the CLI will fail at `snapshot_download` — report the error rather than silently substituting another split. The X-VLA reference uses `libero_goal_no_noops` as the unnorm key, matching OpenVLA's convention.

- [ ] **Step 2: Sanity check**

```bash
PYTHONPATH="" uv run python -c "
import json
d = json.load(open('data/norm_stats/libero_goal.json'))
a = d['libero_goal_no_noops']['action']
print('A:', len(a['q01']))
print('q01:', [round(x,4) for x in a['q01']])
print('q99:', [round(x,4) for x in a['q99']])
print('mask:', a['mask'])
"
```

Expected: `A=7`, mask `[True, True, True, True, True, True, False]`, finite numerical q01/q99.

- [ ] **Step 3: Commit**

```bash
git add data/norm_stats/libero_goal.json
git commit -m "feat(data): compute and persist LIBERO-Goal BOUNDS_Q99 stats"
```

---

## Task 3: New configs for 2-domain smoke

**Files:**
- Create: `configs/data/libero_multidomain.yaml`
- Create: `configs/train/smoke_multidomain.yaml`

- [ ] **Step 1: `configs/data/libero_multidomain.yaml`**

```yaml
type: libero_lerobot_multidomain
seed: 0
download_videos: false
sources:
  - repo_id: lerobot/libero_spatial_image
    stats_path: ${oc.env:LIBERO_STATS_PATH_SPATIAL,data/norm_stats/libero_spatial.json}
    unnorm_key: libero_spatial_no_noops
    fps: 10
    episodes: [0]
    domain_id: 0
    max_samples: 16
    weight: 1.0
  - repo_id: lerobot/libero_goal_image
    stats_path: ${oc.env:LIBERO_STATS_PATH_GOAL,data/norm_stats/libero_goal.json}
    unnorm_key: libero_goal_no_noops
    fps: 10
    episodes: [0]
    domain_id: 1
    max_samples: 16
    weight: 1.0
```

- [ ] **Step 2: `configs/train/smoke_multidomain.yaml`**

```yaml
seed: 0
model:
  num_domains: 2
  hidden_dim: 1536
  num_blocks: 35
  use_grad_checkpoint: true
vision:
  model_name: google/siglip-so400m-patch14-224
language:
  model_name: google/gemma-4-E2B
data:
  type: libero_lerobot_multidomain
  seed: 0
  download_videos: false
  sources:
    - repo_id: lerobot/libero_spatial_image
      stats_path: ${oc.env:LIBERO_STATS_PATH_SPATIAL,data/norm_stats/libero_spatial.json}
      unnorm_key: libero_spatial_no_noops
      fps: 10
      episodes: [0]
      domain_id: 0
      max_samples: 16
      weight: 1.0
    - repo_id: lerobot/libero_goal_image
      stats_path: ${oc.env:LIBERO_STATS_PATH_GOAL,data/norm_stats/libero_goal.json}
      unnorm_key: libero_goal_no_noops
      fps: 10
      episodes: [0]
      domain_id: 1
      max_samples: 16
      weight: 1.0
train:
  batch_size: 1
  lr: 1.0e-4
  soft_lr_coef: 1.0
  weight_decay: 0.01
  max_steps: 2
```

(No commit yet — Task 4 ships these configs together with the dispatcher edit.)

---

## Task 4: `train.py` dispatch for `libero_lerobot_multidomain`

**Files:**
- Modify: `scripts/train.py`

- [ ] **Step 1: Edit `_build_dataloader` to add the new branch**

Replace the existing function body so the dispatcher reads:

```python
def _build_dataloader(cfg: DictConfig, prompt_max_len: int, language_model_name: str):
    data_type = cfg.data.get("type", "libero_synthetic")
    if data_type == "libero_synthetic":
        ds = SyntheticLIBEROBatchDataset(
            length=cfg.data.length, prompt_max_len=prompt_max_len,
        )
        return DataLoader(ds, batch_size=cfg.train.batch_size, collate_fn=ds.collate_fn)
    if data_type == "libero_lerobot_real":
        tok = GemmaPromptTokenizer(model_name=language_model_name, max_len=prompt_max_len)
        ds = LeRobotLiberoDataset(
            repo_id=cfg.data.repo_id,
            stats_path=cfg.data.stats_path,
            unnorm_key=cfg.data.unnorm_key,
            fps=cfg.data.fps,
            tokenizer=tok,
            episodes=list(cfg.data.episodes) if cfg.data.get("episodes") else None,
            download_videos=bool(cfg.data.get("download_videos", False)),
            domain_id=int(cfg.data.get("domain_id", 0)),
            max_samples=cfg.data.get("max_samples", None),
        )
        return DataLoader(
            ds, batch_size=cfg.train.batch_size,
            collate_fn=LeRobotLiberoDataset.collate_fn,
        )
    if data_type == "libero_lerobot_multidomain":
        tok = GemmaPromptTokenizer(model_name=language_model_name, max_len=prompt_max_len)
        children: list = []
        weights: list = []
        for src in cfg.data.sources:
            children.append(LeRobotLiberoDataset(
                repo_id=src.repo_id,
                stats_path=src.stats_path,
                unnorm_key=src.unnorm_key,
                fps=src.fps,
                tokenizer=tok,
                episodes=list(src.episodes) if src.get("episodes") else None,
                download_videos=bool(cfg.data.get("download_videos", False)),
                domain_id=int(src.domain_id),
                max_samples=src.get("max_samples", None),
            ))
            weights.append(float(src.weight))
        ds = WeightedMultiDataset(children, weights, seed=int(cfg.data.get("seed", 0)))
        return DataLoader(
            ds, batch_size=cfg.train.batch_size,
            collate_fn=LeRobotLiberoDataset.collate_fn,
        )
    raise ValueError(f"unknown cfg.data.type: {data_type!r}")
```

Also add the import at the top of `scripts/train.py`:

```python
from vla_project.data.datasets.weighted_multi_dataset import WeightedMultiDataset
```

- [ ] **Step 2: Synthetic + real-data smokes still pass**

```bash
CUDA_VISIBLE_DEVICES=3 PYTHONPATH="" uv run python scripts/train.py configs/train/smoke.yaml
CUDA_VISIBLE_DEVICES=3 PYTHONPATH="" uv run python scripts/train.py configs/train/smoke_real.yaml
```

Expected: both produce two finite losses, no NaN. (Backward-compat check.)

- [ ] **Step 3: Commit**

```bash
git add scripts/train.py configs/data/libero_multidomain.yaml configs/train/smoke_multidomain.yaml
git commit -m "feat(scripts): libero_lerobot_multidomain dispatch + 2-domain smoke configs"
```

---

## Task 5: Multi-domain real smoke

**Files:** none (run-only).

- [ ] **Step 1: Run multi-domain smoke**

```bash
cd /misc/dl00/takaki/X-VLA-Adapter
CUDA_VISIBLE_DEVICES=3 PYTHONPATH="" timeout 600 \
    uv run python scripts/train.py configs/train/smoke_multidomain.yaml 2>&1 | tee /tmp/smoke_multidomain.log
```

Expected: `[train] losses=[<f1>, <f2>]` with two finite floats. The first run downloads `lerobot/libero_goal_image` parquet files (~1-3 min cold). On a warm cache, total wallclock should be similar to the single-domain real smoke (~2 min).

- [ ] **Step 2: Verify the mixer actually drew samples from both domains in the run**

This is hard to verify directly from `losses=[...]` alone. As a sanity check, run a small Python snippet:

```bash
PYTHONPATH="" uv run python -c "
import sys
sys.path.insert(0, 'scripts')
from omegaconf import OmegaConf
from vla_project.data.datasets.lerobot_libero_dataset import LeRobotLiberoDataset
from vla_project.data.datasets.weighted_multi_dataset import WeightedMultiDataset
from vla_project.data.transforms.language import GemmaPromptTokenizer

cfg = OmegaConf.load('configs/train/smoke_multidomain.yaml')
tok = GemmaPromptTokenizer(model_name=cfg.language.model_name, max_len=50)
children = []
weights = []
for src in cfg.data.sources:
    children.append(LeRobotLiberoDataset(
        repo_id=src.repo_id, stats_path=src.stats_path, unnorm_key=src.unnorm_key,
        fps=src.fps, tokenizer=tok,
        episodes=list(src.episodes) if src.get('episodes') else None,
        download_videos=False, domain_id=int(src.domain_id),
        max_samples=src.get('max_samples', None),
    ))
    weights.append(float(src.weight))
mix = WeightedMultiDataset(children, weights, seed=0)
ids = []
it = iter(mix)
for _ in range(20):
    s = next(it)
    ids.append(int(s['domain_id'].item()))
print('drew domain_ids:', ids)
print('counts:', {i: ids.count(i) for i in set(ids)})
"
```

Expected: prints something like `drew domain_ids: [0, 1, 0, 0, 1, 1, 0, 1, ...]` with both 0 and 1 represented (roughly half each given equal weights).

- [ ] **Step 3: Full pytest**

```bash
PYTHONPATH="" uv run pytest -q
```

Expected: all green (Plan-2 totals 55 + 8 new mixer tests = 63 or thereabouts).

No commit — verification only.

---

## Task 6: Push branch + open PR

- [ ] **Step 1: Confirm branch state**

```bash
git status -sb
git log --oneline feat/compute-norm-stats..HEAD
```

The controller should already have created `feat/multi-domain-sampler` branched from `feat/compute-norm-stats` before dispatching Task 1. If not, check out a new branch from `feat/compute-norm-stats`.

- [ ] **Step 2: Push**

```bash
git push -u origin feat/multi-domain-sampler
```

- [ ] **Step 3: Surface PR URL**

PR base: `feat/compute-norm-stats` (rebase to `main` once Plans 1 + 2 merge).
Title: `feat(data): WeightedMultiDataset mixer + 2-domain LIBERO smoke`.
Body should include both `losses=[...]` lines (single-domain backward-compat + new 2-domain) and the test count delta.

---

## Done criteria

- [ ] `uv run pytest -q` passes (full suite, including 8 new `WeightedMultiDataset` tests).
- [ ] `data/norm_stats/libero_goal.json` exists in repo and loads via `load_q99_stats`.
- [ ] `python scripts/train.py configs/train/smoke.yaml` (synthetic) still runs without NaN.
- [ ] `python scripts/train.py configs/train/smoke_real.yaml` (single-domain real) still runs without NaN.
- [ ] `python scripts/train.py configs/train/smoke_multidomain.yaml` (2-domain real) runs ≥ 2 steps without NaN.
- [ ] No edits to `models/`, `policies/`, `robots/`.
- [ ] `WeightedMultiDataset` does not import lerobot directly (only the child dataset does).

## Out of scope (covered by later plans)

- Stats for `libero_object_image` and `libero_10_image` splits (mechanical CLI runs once Plan 3 lands; not blocking).
- LIBERO `last_action_chunk` real prior-chunk fetching (still zeros — defer).
- LoRA Stage 2 (Plan 5).
- Saving the active mixing state in checkpoints (Plan 4 will decide whether to checkpoint sampler RNG state).
