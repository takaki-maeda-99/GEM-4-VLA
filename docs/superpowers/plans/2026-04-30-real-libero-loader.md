# Real LIBERO Loader + Gemma4 Tokenizer + Real-data Smoke (Plan 1 / 10)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace `SyntheticLIBEROBatchDataset` with a real LeRobot-HF based LIBERO loader that yields the project's internal Batch schema with real images, proprio, action chunks, and Gemma4-tokenized prompts; verify a 2-step smoke train passes on dl40 (A100) with real Gemma4-E2B + real SigLIP weights.

**Architecture:** Add `data/transforms/language.py` (Gemma4 prompt tokenizer wrapper with pad-to-max-len), extend `data/normalization.py` with BOUNDS_Q99 helpers + JSON stats loader, add `data/datasets/lerobot_libero_dataset.py` (IterableDataset wrapping `lerobot.datasets.LeRobotDataset` with `delta_timestamps` for action chunks). Wire `scripts/train.py` to dispatch on `cfg.data.type` so synthetic and real both stay reachable. `last_action_chunk` is yielded as zeros (cold-start convention; real prior-chunk fetching is out of scope for Plan 1).

**Tech Stack:** `lerobot` (HF dataset loader), `transformers.AutoTokenizer` (Gemma4-E2B), `huggingface_hub` (info.json fetch), existing torch / torchvision / OmegaConf.

**Repo references (read-only, do not import):**
- `/misc/dl00/takaki/vla-gemma-4/scripts/gemma4/lerobot_libero_loader.py:108-217` — canonical reference for LeRobot LIBERO yield schema and BOUNDS_Q99 normalization (port the algorithm; do not copy the file path or per-timestep dict keys, since this project uses a different internal schema).
- `/misc/dl00/takaki/vla-gemma-4/VLA-Adapter/outputs/LIBERO-Spatial-Pro/dataset_statistics.json` — BOUNDS_Q99 stats (q01 / q99 / mask) under key `libero_spatial_no_noops`. Reuse path from config.
- `/home/takaki/.cache/huggingface/hub/datasets--lerobot--libero_spatial_image/` — already cached LeRobot dataset (432 episodes, 52970 frames, fps=10, images 256×256).
- `docs/architectures/x_vla_adapter.md` — module/tensor contract; the new dataset MUST satisfy `data/schema.py::validate_batch`.

**Hard constraints from `CLAUDE.md`:**
- No LeRobot / RLDS / LIBERO type names leak into `models/` or `policies/`.
- Single internal Batch schema: every key the dataset yields must match `data/schema.py`.
- Fail-fast on shape / key mismatches — no silent reshape, no permissive fallbacks.
- Configs drive paths; no hardcoded HF repo IDs or stats paths in `src/`.

---

## File Structure

**Create:**
- `src/vla_project/data/transforms/language.py`
- `src/vla_project/data/datasets/lerobot_libero_dataset.py`
- `tests/test_language_transform.py`
- `tests/test_normalization_q99.py`
- `tests/test_lerobot_libero_dataset.py`
- `configs/data/libero_real.yaml`
- `configs/train/smoke_real.yaml`

**Modify:**
- `pyproject.toml` (add `lerobot`, `huggingface-hub` deps)
- `src/vla_project/data/normalization.py` (append BOUNDS_Q99 helpers + JSON loader; keep existing `NormalizationStats` / `normalize` / `denormalize` untouched)
- `src/vla_project/data/transforms/__init__.py` (re-export tokenizer class)
- `src/vla_project/data/datasets/__init__.py` (re-export new dataset class)
- `scripts/train.py` (dispatch on `cfg.data.type`)

**Do not modify:** `src/vla_project/models/`, `src/vla_project/policies/`, existing tests.

---

## Task 1: Add `lerobot` + `huggingface-hub` runtime deps

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Add deps via uv**

```bash
cd /misc/dl00/takaki/X-VLA-Adapter
PYTHONPATH="" uv add lerobot huggingface-hub
```

- [ ] **Step 2: Verify import works**

```bash
PYTHONPATH="" uv run python -c "import lerobot; from lerobot.datasets.lerobot_dataset import LeRobotDataset; from huggingface_hub import hf_hub_download; print('ok', lerobot.__version__)"
```

Expected: `ok <version>` line, no traceback.

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "feat(deps): add lerobot + huggingface-hub for real LIBERO loader"
```

---

## Task 2: Baseline synthetic smoke on dl40 (no code changes)

This is verification, not feature work — proves the existing `configs/train/smoke.yaml` runs end-to-end on dl40 with real Gemma4-E2B + real SigLIP before we change the data path.

**Files:** none changed.

- [ ] **Step 1: Pre-flight environment check**

```bash
nvidia-smi --query-gpu=name,memory.free --format=csv
echo "---"
ls /home/takaki/.cache/huggingface/hub/ | grep -E "gemma-4-E2B|siglip" || echo "(siglip-google not cached, will download ~1.6 GB)"
```

Expected: at least one A100 (40GB+) row in `nvidia-smi`. If `google/siglip-so400m-patch14-224` is missing, the next step downloads it automatically.

- [ ] **Step 2: Run synthetic smoke**

```bash
cd /misc/dl00/takaki/X-VLA-Adapter
PYTHONPATH="" uv run python scripts/train.py configs/train/smoke.yaml
```

Expected output: `[train] device=cuda dtype=torch.bfloat16` then `[train] losses=[<f1>, <f2>]` with two finite floats. No NaN, no traceback. End-to-end wallclock should be < 5 min on A100 (first run includes SigLIP download).

- [ ] **Step 3: Record baseline loss for sanity comparison later**

In your shell, capture stdout to a file for reference:

```bash
PYTHONPATH="" uv run python scripts/train.py configs/train/smoke.yaml 2>&1 | tee /tmp/smoke_synthetic_baseline.log
```

No commit — this is verification only.

---

## Task 3: Gemma4 prompt tokenizer (`data/transforms/language.py`)

**Files:**
- Create: `src/vla_project/data/transforms/language.py`
- Create: `tests/test_language_transform.py`
- Modify: `src/vla_project/data/transforms/__init__.py`

- [ ] **Step 1: Write failing test**

`tests/test_language_transform.py`:

```python
import pytest
import torch

from vla_project.data import constants as C
from vla_project.data.transforms.language import GemmaPromptTokenizer


@pytest.fixture(scope="module")
def tok() -> GemmaPromptTokenizer:
    return GemmaPromptTokenizer(model_name="google/gemma-4-E2B", max_len=C.DEFAULT_PROMPT_MAX_LEN)


def test_short_prompt_padded_right(tok: GemmaPromptTokenizer) -> None:
    out = tok("pick up the red block")
    assert out["input_ids"].shape == (C.DEFAULT_PROMPT_MAX_LEN,)
    assert out["attention_mask"].shape == (C.DEFAULT_PROMPT_MAX_LEN,)
    assert out["input_ids"].dtype == torch.long
    assert out["attention_mask"].dtype == torch.long
    # Padding lives at the right end
    assert out["attention_mask"][0].item() == 1
    assert out["attention_mask"][-1].item() == 0


def test_long_prompt_truncated(tok: GemmaPromptTokenizer) -> None:
    long = " ".join(["block"] * 200)
    out = tok(long)
    assert out["input_ids"].shape == (C.DEFAULT_PROMPT_MAX_LEN,)
    # When truncated, every position is real (mask all ones)
    assert out["attention_mask"].sum().item() == C.DEFAULT_PROMPT_MAX_LEN


def test_batch_call_stacks(tok: GemmaPromptTokenizer) -> None:
    batch = tok.batch(["pick the red block", "stack the blue cube on the green plate"])
    assert batch["input_ids"].shape == (2, C.DEFAULT_PROMPT_MAX_LEN)
    assert batch["attention_mask"].shape == (2, C.DEFAULT_PROMPT_MAX_LEN)
```

- [ ] **Step 2: Run test, expect FAIL**

```bash
PYTHONPATH="" uv run pytest tests/test_language_transform.py -v
```

Expected: `ImportError: cannot import name 'GemmaPromptTokenizer'`.

- [ ] **Step 3: Implement**

`src/vla_project/data/transforms/language.py`:

```python
"""Gemma4 prompt tokenizer wrapper.

Tokenizes a single language instruction or a list of instructions to a fixed
length (`prompt_max_len`), right-padding with the tokenizer's pad token.
Returns torch tensors keyed `input_ids`, `attention_mask` so the result drops
straight into the project's internal Batch schema.

The tokenizer is loaded via `AutoTokenizer.from_pretrained` and is **not**
fine-tuned. It is instantiated once per dataset (or process) and reused.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch

from vla_project.data import constants as C


@dataclass
class _Tokenized:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor

    def __getitem__(self, key: str) -> torch.Tensor:
        return getattr(self, key)


class GemmaPromptTokenizer:
    def __init__(
        self,
        model_name: str = "google/gemma-4-E2B",
        max_len: int = C.DEFAULT_PROMPT_MAX_LEN,
        _tokenizer=None,
    ) -> None:
        self.max_len = max_len
        if _tokenizer is not None:
            self._tok = _tokenizer
        else:
            from transformers import AutoTokenizer
            self._tok = AutoTokenizer.from_pretrained(model_name)
        if self._tok.pad_token_id is None:
            # Gemma4 tokenizer ships a pad token; fall back to eos defensively.
            self._tok.pad_token = self._tok.eos_token

    def __call__(self, text: str) -> Dict[str, torch.Tensor]:
        enc = self._tok(
            text,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        # AutoTokenizer returns [1, L]; squeeze to [L].
        return {
            "input_ids": enc["input_ids"].squeeze(0).to(torch.long),
            "attention_mask": enc["attention_mask"].squeeze(0).to(torch.long),
        }

    def batch(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        enc = self._tok(
            texts,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].to(torch.long),
            "attention_mask": enc["attention_mask"].to(torch.long),
        }
```

`src/vla_project/data/transforms/__init__.py` — append:

```python
from vla_project.data.transforms.language import GemmaPromptTokenizer  # noqa: F401
```

- [ ] **Step 4: Run, expect PASS**

```bash
PYTHONPATH="" uv run pytest tests/test_language_transform.py -v
```

Expected: 3 passed. (First call downloads the Gemma4 tokenizer if not cached — should already be cached on dl40.)

- [ ] **Step 5: Commit**

```bash
git add src/vla_project/data/transforms/language.py \
        src/vla_project/data/transforms/__init__.py \
        tests/test_language_transform.py
git commit -m "feat(data): GemmaPromptTokenizer wrapper for prompt input_ids"
```

---

## Task 4: BOUNDS_Q99 action normalization + stats JSON loader

**Files:**
- Modify: `src/vla_project/data/normalization.py` (append, do not change existing functions)
- Create: `tests/test_normalization_q99.py`

- [ ] **Step 1: Write failing test**

`tests/test_normalization_q99.py`:

```python
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from vla_project.data.normalization import (
    Q99Stats,
    load_q99_stats,
    normalize_action_q99,
)


def _write_stats(tmp_path: Path) -> Path:
    payload = {
        "libero_spatial_no_noops": {
            "action": {
                "q01": [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0,  0.0],
                "q99": [ 1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0],
                "mask": [True, True, True, True, True, True, False],
            }
        }
    }
    p = tmp_path / "stats.json"
    p.write_text(json.dumps(payload))
    return p


def test_load_q99_stats_round_trip(tmp_path: Path) -> None:
    p = _write_stats(tmp_path)
    stats = load_q99_stats(p, unnorm_key="libero_spatial_no_noops")
    assert isinstance(stats, Q99Stats)
    assert stats.q01.shape == (7,)
    assert stats.q99.shape == (7,)
    assert stats.mask.shape == (7,)
    assert stats.mask.dtype == torch.bool
    assert stats.mask[-1].item() is False  # gripper dim unchanged


def test_normalize_action_q99_clips_to_unit(tmp_path: Path) -> None:
    p = _write_stats(tmp_path)
    stats = load_q99_stats(p, unnorm_key="libero_spatial_no_noops")
    raw = torch.tensor([
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5],   # midpoint of q01..q99 -> 0
        [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 1.0],   # above q99 -> clipped to 1
        [-2.0, -2.0, -2.0, -2.0, -2.0, -2.0, 0.0],  # below q01 -> -1
    ], dtype=torch.float32)
    out = normalize_action_q99(raw, stats)
    assert out.shape == raw.shape
    assert out.dtype == torch.float32
    # First 6 dims (mask=True) clipped into [-1, 1]
    assert torch.allclose(out[0, :6], torch.zeros(6), atol=1e-6)
    assert torch.allclose(out[1, :6], torch.ones(6), atol=1e-6)
    assert torch.allclose(out[2, :6], -torch.ones(6), atol=1e-6)
    # Last dim (mask=False, gripper) untouched
    assert out[0, 6].item() == pytest.approx(0.5)
    assert out[1, 6].item() == pytest.approx(1.0)
```

- [ ] **Step 2: Run, expect FAIL**

```bash
PYTHONPATH="" uv run pytest tests/test_normalization_q99.py -v
```

Expected: `ImportError: cannot import name 'Q99Stats'`.

- [ ] **Step 3: Implement (append to existing `normalization.py`)**

`src/vla_project/data/normalization.py` — append below the existing code:

```python
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Union


@dataclass
class Q99Stats:
    """Per-dim BOUNDS_Q99 stats for action normalization (X-VLA / OpenVLA convention).

    For dims where ``mask[i] == True``, the dim is rescaled to [-1, 1] using
    q01 / q99. Where ``mask[i] == False`` (typically the binary gripper), the
    value is passed through unchanged.
    """

    q01: torch.Tensor   # [A]
    q99: torch.Tensor   # [A]
    mask: torch.Tensor  # [A] bool


def load_q99_stats(path: Union[str, Path], unnorm_key: str) -> Q99Stats:
    """Load BOUNDS_Q99 stats from a `dataset_statistics.json` produced by the
    OpenVLA / VLA-Adapter pipelines.

    The JSON is keyed by dataset name; each entry has an ``action`` block with
    ``q01``, ``q99``, and (optionally) ``mask`` lists of length A.
    """
    payload = json.loads(Path(path).read_text())
    if unnorm_key not in payload:
        raise KeyError(
            f"unnorm_key {unnorm_key!r} not in {path}; available: {list(payload.keys())}"
        )
    action = payload[unnorm_key]["action"]
    q01 = torch.as_tensor(action["q01"], dtype=torch.float32)
    q99 = torch.as_tensor(action["q99"], dtype=torch.float32)
    if "mask" in action:
        mask = torch.as_tensor(action["mask"], dtype=torch.bool)
    else:
        mask = torch.ones_like(q01, dtype=torch.bool)
    if not (q01.shape == q99.shape == mask.shape):
        raise ValueError(
            f"q01/q99/mask shape mismatch: {q01.shape}, {q99.shape}, {mask.shape}"
        )
    return Q99Stats(q01=q01, q99=q99, mask=mask)


def normalize_action_q99(action_raw: torch.Tensor, stats: Q99Stats) -> torch.Tensor:
    """Forward BOUNDS_Q99 normalization. Inverse of the eval-time denormalize.

    For ``mask=True`` dims: rescale (q01, q99) -> (-1, 1) and clip.
    For ``mask=False`` dims: passthrough.

    Args:
        action_raw: [..., A]
        stats: Q99Stats with shape [A]

    Returns:
        Tensor of same shape and dtype as ``action_raw``.
    """
    if action_raw.shape[-1] != stats.q01.shape[0]:
        raise ValueError(
            f"action last dim {action_raw.shape[-1]} != stats dim {stats.q01.shape[0]}"
        )
    q01 = stats.q01.to(action_raw.dtype).to(action_raw.device)
    q99 = stats.q99.to(action_raw.dtype).to(action_raw.device)
    mask = stats.mask.to(action_raw.device)
    denom = (q99 - q01).clamp_min(1e-8)
    norm = (2.0 * (action_raw - q01) / denom - 1.0).clamp(-1.0, 1.0)
    return torch.where(mask, norm, action_raw)
```

- [ ] **Step 4: Run, expect PASS**

```bash
PYTHONPATH="" uv run pytest tests/test_normalization_q99.py -v
PYTHONPATH="" uv run pytest tests/test_normalization.py -v   # existing tests still green
```

Expected: both green.

- [ ] **Step 5: Commit**

```bash
git add src/vla_project/data/normalization.py tests/test_normalization_q99.py
git commit -m "feat(data): BOUNDS_Q99 action normalization + JSON stats loader"
```

---

## Task 5: `LeRobotLiberoDataset` (real loader)

**Files:**
- Create: `src/vla_project/data/datasets/lerobot_libero_dataset.py`
- Create: `tests/test_lerobot_libero_dataset.py`
- Modify: `src/vla_project/data/datasets/__init__.py`

**Design notes (locked in):**
- IterableDataset (matches reference loader; LIBERO is large, infinite training).
- `delta_timestamps={"action": [i / fps for i in range(action_chunk_len)]}` produces the future chunk only.
- `last_action_chunk` yielded as `torch.zeros(H, A)` — cold-start convention. Adding real prior-chunk lookup is out of scope (Plan 1) and noted in the dataset docstring.
- `domain_id = 0` (single-domain). Multi-domain comes in Plan 3.
- `proprio` returned raw (not normalized); the head's `proprio_proj` is a learned linear, matching reference behavior.
- Image pipeline: LeRobot tensor `(3, 256, 256) float [0,1]` → bilinear resize 224 → SigLIP normalize.
- Prompt: `meta.tasks` reverse-mapped at init time; tokenized per-sample with `GemmaPromptTokenizer`.
- `target_action` = LeRobot raw `action` chunk passed through `normalize_action_q99`.
- `action_mask` = all True (LeRobot guarantees full-length chunks via tolerance; we fail-fast on length mismatch).

- [ ] **Step 1: Write failing test (offline-friendly: stub LeRobotDataset)**

`tests/test_lerobot_libero_dataset.py`:

```python
"""Offline test for LeRobotLiberoDataset.

We stub `lerobot.datasets.lerobot_dataset.LeRobotDataset` so the test runs
without network or HF cache access. The stub yields tensors with the same
shapes / dtypes the real loader returns. The test asserts the dataset
emits batches that satisfy `data/schema.py::validate_batch`.
"""
from pathlib import Path
from typing import Dict, List

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from vla_project.data import constants as C
from vla_project.data.schema import validate_batch
from vla_project.data.transforms.language import GemmaPromptTokenizer


class _StubMeta:
    def __init__(self) -> None:
        # Mimic the dict form the real loader's resolver supports.
        self.tasks = {0: "pick the red block"}


class _StubLeRobotDataset:
    def __init__(self, *_, **__) -> None:
        self.meta = _StubMeta()

    def __len__(self) -> int:
        return 4

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "observation.images.image":       torch.rand(3, 256, 256),
            "observation.images.wrist_image": torch.rand(3, 256, 256),
            "observation.state":              torch.randn(C.PROPRIO_DIM),
            "action":                         torch.randn(C.ACTION_CHUNK_LEN, C.ACTION_DIM),
            "task_index":                     torch.tensor(0, dtype=torch.long),
        }


class _StubTokenizer:
    """Avoids the network on `AutoTokenizer.from_pretrained`."""

    pad_token_id = 0
    eos_token = "<eos>"
    pad_token = "<pad>"

    def __call__(self, text, **kw):
        L = kw.get("max_length", C.DEFAULT_PROMPT_MAX_LEN)
        if isinstance(text, str):
            ids = torch.zeros(1, L, dtype=torch.long)
            mask = torch.zeros(1, L, dtype=torch.long)
            mask[0, : min(len(text.split()), L)] = 1
            return {"input_ids": ids, "attention_mask": mask}
        # batch
        out_ids = torch.zeros(len(text), L, dtype=torch.long)
        out_mask = torch.zeros(len(text), L, dtype=torch.long)
        for i, t in enumerate(text):
            out_mask[i, : min(len(t.split()), L)] = 1
        return {"input_ids": out_ids, "attention_mask": out_mask}


@pytest.fixture
def stats_path(tmp_path: Path) -> Path:
    import json
    payload = {
        "libero_spatial_no_noops": {
            "action": {
                "q01": [-1.0] * C.ACTION_DIM,
                "q99": [ 1.0] * C.ACTION_DIM,
                "mask": [True, True, True, True, True, True, False],
            }
        }
    }
    p = tmp_path / "stats.json"
    p.write_text(json.dumps(payload))
    return p


def test_yields_valid_batch(monkeypatch, stats_path: Path) -> None:
    from vla_project.data.datasets import lerobot_libero_dataset as M

    monkeypatch.setattr(M, "_LeRobotDatasetCls", _StubLeRobotDataset)

    tok = GemmaPromptTokenizer(model_name=None, _tokenizer=_StubTokenizer())
    ds = M.LeRobotLiberoDataset(
        repo_id="lerobot/libero_spatial_image",
        stats_path=str(stats_path),
        unnorm_key="libero_spatial_no_noops",
        fps=10,
        tokenizer=tok,
        episodes=[0],
        download_videos=False,
        domain_id=0,
        max_samples=4,
    )
    dl = DataLoader(ds, batch_size=2, collate_fn=M.LeRobotLiberoDataset.collate_fn)
    batch = next(iter(dl))
    validate_batch(batch)
    assert batch["domain_id"].shape == (2,)
    assert batch["scene_image"].shape == (2, 3, C.SIGLIP_IMAGE_SIZE, C.SIGLIP_IMAGE_SIZE)
    assert batch["wrist_image"].shape == (2, 3, C.SIGLIP_IMAGE_SIZE, C.SIGLIP_IMAGE_SIZE)
    assert batch["proprio"].shape == (2, C.PROPRIO_DIM)
    assert batch["last_action_chunk"].shape == (2, C.ACTION_CHUNK_LEN, C.ACTION_DIM)
    assert batch["target_action"].shape == (2, C.ACTION_CHUNK_LEN, C.ACTION_DIM)
    assert batch["action_mask"].shape == (2, C.ACTION_CHUNK_LEN)
    assert batch["prompt_input_ids"].shape == (2, C.DEFAULT_PROMPT_MAX_LEN)
    # Cold-start convention: last_action_chunk is zeros at training time.
    assert torch.all(batch["last_action_chunk"] == 0.0)
    # Target actions clipped to [-1, 1] (mask=True dims).
    assert batch["target_action"][:, :, :6].abs().max().item() <= 1.0 + 1e-6
```

- [ ] **Step 2: Run, expect FAIL**

```bash
PYTHONPATH="" uv run pytest tests/test_lerobot_libero_dataset.py -v
```

Expected: `ModuleNotFoundError` on `vla_project.data.datasets.lerobot_libero_dataset`.

- [ ] **Step 3: Implement**

`src/vla_project/data/datasets/lerobot_libero_dataset.py`:

```python
"""LeRobot-HF based LIBERO step-level dataset.

Yields the project's internal Batch schema (see `data/schema.py`). Wraps
`lerobot.datasets.LeRobotDataset` with `delta_timestamps` so each yielded
sample contains an action chunk of length `ACTION_CHUNK_LEN`. Images are
resized to SigLIP's 224×224, normalized by SigLIP statistics. Action chunks
are normalized with BOUNDS_Q99 stats loaded from JSON. Prompts are tokenized
with the project's `GemmaPromptTokenizer`.

Single-domain only (Plan 1). `last_action_chunk` is zeros (cold-start; real
prior-chunk fetching is Plan 3 / future work).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Union

import torch
from torch.utils.data import IterableDataset

from vla_project.data import constants as C
from vla_project.data.normalization import Q99Stats, load_q99_stats, normalize_action_q99
from vla_project.data.transforms.image import SiglipImageTransform
from vla_project.data.transforms.language import GemmaPromptTokenizer


# Indirection so tests can monkey-patch without importing lerobot at import time.
def _default_lerobot_cls():
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    return LeRobotDataset


_LeRobotDatasetCls = None  # populated lazily; tests can override via monkeypatch


class LeRobotLiberoDataset(IterableDataset):
    def __init__(
        self,
        repo_id: str,
        stats_path: Union[str, Path],
        unnorm_key: str,
        fps: int,
        tokenizer: GemmaPromptTokenizer,
        episodes: Optional[List[int]] = None,
        action_chunk_len: int = C.ACTION_CHUNK_LEN,
        download_videos: bool = True,
        domain_id: int = 0,
        max_samples: Optional[int] = None,
    ) -> None:
        super().__init__()
        global _LeRobotDatasetCls
        if _LeRobotDatasetCls is None:
            _LeRobotDatasetCls = _default_lerobot_cls()
        delta = {"action": [i / fps for i in range(action_chunk_len)]}
        self.ds = _LeRobotDatasetCls(
            repo_id,
            delta_timestamps=delta,
            episodes=episodes,
            download_videos=download_videos,
        )
        self.action_chunk_len = action_chunk_len
        self.domain_id = int(domain_id)
        self.max_samples = max_samples
        self.stats: Q99Stats = load_q99_stats(stats_path, unnorm_key)
        self.image_tx = SiglipImageTransform(size=C.SIGLIP_IMAGE_SIZE, training=False)
        self.tokenizer = tokenizer
        self._task_idx_to_str: Dict[int, str] = self._build_task_map()

    def _build_task_map(self) -> Dict[int, str]:
        out: Dict[int, str] = {}
        tasks = self.ds.meta.tasks
        if hasattr(tasks, "iterrows"):
            for task_str, row in tasks.iterrows():
                out[int(row["task_index"])] = str(task_str).strip()
        elif isinstance(tasks, dict):
            for k, v in tasks.items():
                out[int(k)] = str(v).strip()
        elif isinstance(tasks, (list, tuple)):
            for i, v in enumerate(tasks):
                out[i] = str(v).strip()
        else:
            raise TypeError(f"unsupported tasks meta type: {type(tasks)!r}")
        return out

    def _resize_image(self, lerobot_img: torch.Tensor) -> torch.Tensor:
        if lerobot_img.shape[0] != 3:
            raise ValueError(f"expected (3, H, W), got {tuple(lerobot_img.shape)}")
        return self.image_tx(lerobot_img)

    def _sample_to_batch_item(self, sample: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        scene = self._resize_image(sample["observation.images.image"])
        wrist = self._resize_image(sample["observation.images.wrist_image"])
        proprio = sample["observation.state"].to(torch.float32)
        if proprio.shape != (C.PROPRIO_DIM,):
            raise ValueError(f"proprio shape {tuple(proprio.shape)} != ({C.PROPRIO_DIM},)")
        action_raw = sample["action"].to(torch.float32)
        if action_raw.shape != (self.action_chunk_len, C.ACTION_DIM):
            raise ValueError(
                f"action shape {tuple(action_raw.shape)} != "
                f"({self.action_chunk_len}, {C.ACTION_DIM})"
            )
        target_action = normalize_action_q99(action_raw, self.stats)

        task_idx_t = sample["task_index"]
        task_idx = int(task_idx_t.item()) if torch.is_tensor(task_idx_t) else int(task_idx_t)
        prompt_text = self._task_idx_to_str.get(task_idx, "")
        prompt = self.tokenizer(prompt_text)

        return {
            "domain_id": torch.tensor(self.domain_id, dtype=torch.long),
            "scene_image": scene,
            "wrist_image": wrist,
            "prompt_input_ids": prompt["input_ids"],
            "prompt_attention_mask": prompt["attention_mask"],
            "proprio": proprio,
            "last_action_chunk": torch.zeros(
                self.action_chunk_len, C.ACTION_DIM, dtype=torch.float32
            ),
            "target_action": target_action,
            "action_mask": torch.ones(self.action_chunk_len, dtype=torch.bool),
        }

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        emitted = 0
        # Single-pass iteration. Trainer calls iter(dataloader) again to restart.
        for i in range(len(self.ds)):
            if self.max_samples is not None and emitted >= self.max_samples:
                return
            sample = self.ds[i]
            if sample["action"].shape[0] != self.action_chunk_len:
                # delta_timestamps near episode end may yield a short chunk; skip.
                continue
            yield self._sample_to_batch_item(sample)
            emitted += 1

    @staticmethod
    def collate_fn(samples: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        keys = samples[0].keys()
        return {k: torch.stack([s[k] for s in samples]) for k in keys}
```

`src/vla_project/data/datasets/__init__.py` — append:

```python
from vla_project.data.datasets.lerobot_libero_dataset import LeRobotLiberoDataset  # noqa: F401
```

- [ ] **Step 4: Run, expect PASS**

```bash
PYTHONPATH="" uv run pytest tests/test_lerobot_libero_dataset.py -v
```

Expected: 1 passed (uses monkey-patched stub; no network).

- [ ] **Step 5: Commit**

```bash
git add src/vla_project/data/datasets/lerobot_libero_dataset.py \
        src/vla_project/data/datasets/__init__.py \
        tests/test_lerobot_libero_dataset.py
git commit -m "feat(data): LeRobotLiberoDataset (real LIBERO via lerobot HF)"
```

---

## Task 6: `train.py` dispatch + real-data configs

**Files:**
- Modify: `scripts/train.py`
- Create: `configs/data/libero_real.yaml`
- Create: `configs/train/smoke_real.yaml`

- [ ] **Step 1: Add real-data config files**

`configs/data/libero_real.yaml`:

```yaml
type: libero_lerobot_real
repo_id: lerobot/libero_spatial_image
stats_path: /misc/dl00/takaki/vla-gemma-4/VLA-Adapter/outputs/LIBERO-Spatial-Pro/dataset_statistics.json
unnorm_key: libero_spatial_no_noops
fps: 10
episodes: [0]            # single episode for smoke; remove key for full split
download_videos: false   # parquet-only LeRobot variant has no video files
domain_id: 0
max_samples: 16
```

`configs/train/smoke_real.yaml`:

```yaml
seed: 0
model:
  num_domains: 1
  hidden_dim: 1536
  num_blocks: 35
  use_grad_checkpoint: true
vision:
  model_name: google/siglip-so400m-patch14-224
language:
  model_name: google/gemma-4-E2B
data:
  type: libero_lerobot_real
  repo_id: lerobot/libero_spatial_image
  stats_path: /misc/dl00/takaki/vla-gemma-4/VLA-Adapter/outputs/LIBERO-Spatial-Pro/dataset_statistics.json
  unnorm_key: libero_spatial_no_noops
  fps: 10
  episodes: [0]
  download_videos: false
  domain_id: 0
  max_samples: 16
train:
  batch_size: 1
  lr: 1.0e-4
  soft_lr_coef: 1.0
  weight_decay: 0.01
  max_steps: 2
```

- [ ] **Step 2: Modify `scripts/train.py` to dispatch on `cfg.data.type`**

Replace the dataset construction block. The new file:

```python
"""Thin training entrypoint. Heavy lifting lives in vla_project.training.trainer."""
from pathlib import Path
from typing import Iterable

import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from vla_project.data.datasets.libero_dataset import SyntheticLIBEROBatchDataset
from vla_project.data.datasets.lerobot_libero_dataset import LeRobotLiberoDataset
from vla_project.data.transforms.language import GemmaPromptTokenizer
from vla_project.models.language.gemma4_wrapper import Gemma4Wrapper
from vla_project.models.vision.siglip import SigLIPEncoder
from vla_project.models.vla_policy import VLAPolicy, VLAPolicyConfig
from vla_project.training.optim import build_optimizer
from vla_project.training.trainer import Trainer, TrainerConfig
from vla_project.utils.seed import set_seed


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
    raise ValueError(f"unknown cfg.data.type: {data_type!r}")


def main(cfg_path: str) -> None:
    cfg = OmegaConf.load(cfg_path)
    set_seed(cfg.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    print(f"[train] device={device} dtype={dtype}")

    policy_cfg = VLAPolicyConfig(**cfg.model)
    vision = SigLIPEncoder(model_name=cfg.vision.model_name)
    gemma = Gemma4Wrapper(model_name=cfg.language.model_name, freeze=True)
    policy = VLAPolicy(policy_cfg, vision, gemma).to(device).to(dtype)

    dl = _build_dataloader(
        cfg, prompt_max_len=policy_cfg.prompt_max_len,
        language_model_name=cfg.language.model_name,
    )

    optim = build_optimizer(
        policy, lr=cfg.train.lr,
        soft_lr_coef=cfg.train.soft_lr_coef, weight_decay=cfg.train.weight_decay,
    )
    trainer = Trainer(policy, optim, TrainerConfig(max_steps=cfg.train.max_steps))
    losses = trainer.fit(dl)
    print(f"[train] losses={losses}")


if __name__ == "__main__":
    import sys
    main(sys.argv[1])
```

- [ ] **Step 3: Run existing synthetic config — must still work**

```bash
PYTHONPATH="" uv run python scripts/train.py configs/train/smoke.yaml
```

Expected: same baseline behavior as Task 2 — `[train] losses=[<f1>, <f2>]`, no NaN. This proves the dispatch did not break the synthetic path.

- [ ] **Step 4: Commit**

```bash
git add scripts/train.py configs/data/libero_real.yaml configs/train/smoke_real.yaml
git commit -m "feat(scripts): dispatch train.py on cfg.data.type; add real-libero configs"
```

---

## Task 7: Real-data smoke train run + final verification

**Files:** none changed (run-only).

- [ ] **Step 1: Run real-data smoke**

```bash
cd /misc/dl00/takaki/X-VLA-Adapter
PYTHONPATH="" uv run python scripts/train.py configs/train/smoke_real.yaml 2>&1 | tee /tmp/smoke_real.log
```

Expected output (in order):
1. `[train] device=cuda dtype=torch.bfloat16`
2. (LeRobotDataset may print download progress for episode 0 of `lerobot/libero_spatial_image`; the parquet is already cached so this should be near-instant.)
3. `[train] losses=[<f1>, <f2>]` — two finite floats, no NaN.

If `[train] losses=[nan, ...]` appears: stop and inspect `/tmp/smoke_real.log` for the first numeric anomaly. Common causes are bad stats path (silently returning all-zero q01/q99) or proprio dtype mismatch — both fail-fast with errors in earlier tasks, so this should not happen.

- [ ] **Step 2: Run full pytest**

```bash
PYTHONPATH="" uv run pytest -q
```

Expected: all tests pass, including the three new test files.

- [ ] **Step 3: Commit log artifacts (none) — open PR**

```bash
git checkout -b feat/real-libero-loader
git push -u origin feat/real-libero-loader
```

PR title: `feat(data): real LIBERO loader (LeRobot HF) + Gemma4 prompt tokenizer`

PR body should reference:
- Plan: `docs/superpowers/plans/2026-04-30-real-libero-loader.md`
- Smoke evidence: paste the two `[train] losses=[...]` lines from `/tmp/smoke_real.log`
- Test count delta: `before=N, after=N+3`

---

## Done criteria

- [ ] `uv run pytest -q` passes (all old + 3 new tests).
- [ ] `python scripts/train.py configs/train/smoke.yaml` still runs (synthetic path unbroken).
- [ ] `python scripts/train.py configs/train/smoke_real.yaml` runs ≥ 2 steps without NaN on a real LIBERO sample (real Gemma4-E2B + real SigLIP weights, on dl40 A100).
- [ ] `models/`, `policies/`, `robots/` files unchanged.
- [ ] No hardcoded HF repo IDs or stats paths in `src/`.

## Out of scope (covered by later plans)

- Real norm-stats computation tool (Plan 2).
- Multi-domain weighted sampler with non-zero `last_action_chunk` (Plan 3).
- Checkpoint save/load with norm_stats embedded (Plan 4).
- LoRA (Plan 5), policies/eval/robots (Plans 6–8).
