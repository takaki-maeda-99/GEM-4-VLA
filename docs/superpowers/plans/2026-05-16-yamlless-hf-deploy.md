# Yaml-less HF-driven Deployment Server — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor the inference server to run from `--checkpoint <local|HF-id>` alone. Delete the deploy yaml. Server returns model-native q99-denormed action chunks (optionally post-processed by `<ckpt_dir>/post_process.py`). Contract translation moves to clients.

**Architecture:** `DomainAdapter` + `DeployConfig` are demolished. Salvageable logic (JPEG decode, F1 image sanity, F3 proprio OOD, q99 denorm with mask, NaN guards) is split into focused modules in `src/vla_project/deployment/`. `ModelRuntime` gains an `is_local` flag and optional `post_process_fn`. A new `/metadata` endpoint replaces `/admin/schema`. Training-side `checkpoint.py` learns to write a `meta.native_action` block.

**Tech Stack:** Python 3.10+, FastAPI, pydantic v2, PyTorch, NumPy, pytest, huggingface_hub, uv.

**Spec:** [`docs/superpowers/specs/2026-05-16-yamlless-hf-deploy-design.md`](../specs/2026-05-16-yamlless-hf-deploy-design.md)

---

## File map

### Create
- `src/vla_project/deployment/wire_io.py` — JPEG decode + F1 image sanity + proprio normalize + F3 OOD + q99 denorm + NaN guards
- `src/vla_project/deployment/post_process_loader.py` — `<ckpt_dir>/post_process.py` loader with trust gating
- `src/vla_project/deployment/startup_validation.py` — non-contract startup checks
- `src/vla_project/deployment/metadata.py` — `/metadata` response builder
- `tools/backfill_meta_native_action.py` — one-off backfill of `meta.native_action` for existing checkpoints
- `tests/test_wire_io_jpeg.py`
- `tests/test_wire_io_denorm.py`
- `tests/test_wire_io_proprio_ood.py`
- `tests/test_post_process_loader.py`
- `tests/test_startup_validation.py`
- `tests/test_metadata_response.py`
- `tests/test_runtime_post_process.py`
- `tests/test_inference_server_yamlless.py`
- `tests/test_checkpoint_native_action.py`
- `tests/test_backfill_native_action.py`

### Modify
- `src/vla_project/deployment/runtime.py` — `_resolve_ckpt_dir` returns `(path, is_local)`; `ModelRuntime` stores `is_local` and loads `post_process_fn`
- `src/vla_project/deployment/inference_server.py` — full rewrite of `build_app` (no deploy yaml), `/metadata` route, predictor wiring via new modules
- `scripts/serve.py` — new CLI shape
- `src/vla_project/training/checkpoint.py` — write `meta.native_action` from `cfg.data.native_action` when present
- `src/vla_project/deployment/predictors/xvla_adapter.py` — drop docstring references to `DomainAdapter`
- `src/vla_project/deployment/predictors/hold_position.py` — accept explicit `chunk_len`, `action_dim`, `gripper_midpoint` from CLI rather than `DeployConfig`
- `README.md` — replace launch examples, add `--trust-checkpoint-code` note

### Delete
- `configs/deploy/_template.yaml`
- `configs/deploy/so101_v46.yaml`
- `configs/deploy/v36_libero_spatial.yaml`
- `configs/deploy/mimicrec_pairing_example.yaml`
- `src/vla_project/deployment/domain_adapter.py`
- `tests/test_domain_adapter.py`
- `tests/test_validation_image_sanity.py` (superseded by `test_wire_io_jpeg.py`)
- `tests/test_validation_proprio.py` (superseded by `test_wire_io_proprio_ood.py`)
- `tests/test_admin_schema.py` (superseded by `test_metadata_response.py`)
- `tests/test_serve_smoke.py` (superseded by `test_inference_server_yamlless.py`)
- `tests/test_inference_server_minimal.py` (superseded by `test_inference_server_yamlless.py`)
- `tests/test_validation_prompt.py` — KEEP if it tests prompt tokenizer behavior independently; DELETE if it goes through DomainAdapter (check at Task 13)

---

## Phase A — New modules (purely additive, all existing tests stay green)

### Task 1: `wire_io.q99_denorm` — denorm with mask

**Files:**
- Test: `tests/test_wire_io_denorm.py`
- Create: `src/vla_project/deployment/wire_io.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_wire_io_denorm.py
"""q99 denorm with mask: dims where mask=True are q99-inverse;
dims where mask=False pass through unchanged.

Mirrors data/normalization.py:denormalize_action_q99 (the source of
truth) — this test exists because the deployment side now owns its own
copy of the logic, decoupled from the dataset module.
"""
from __future__ import annotations

import numpy as np
import pytest

from vla_project.deployment.wire_io import q99_denorm_with_mask


def _stats(action_dim: int = 4) -> dict:
    """Synthetic stats: q01 = -1, q99 = +1 → span=2; any mean/std/min/max ok."""
    return {
        "q01":  [-1.0] * action_dim,
        "q99":  [+1.0] * action_dim,
        "mask": [True, True, True, False],
        "mean": [0.0] * action_dim,
        "std":  [1.0] * action_dim,
        "min":  [-1.0] * action_dim,
        "max":  [+1.0] * action_dim,
    }


def test_denorm_masked_dims_are_q99_inverse():
    stats = _stats()
    # action_norm in [-1, +1] → expected in [-1, +1] (since q01=-1, q99=+1)
    action_norm = np.array([[0.0, 0.5, -0.5, 0.0]], dtype=np.float32)
    out = q99_denorm_with_mask(action_norm, stats)
    np.testing.assert_allclose(out[0, :3], [0.0, 0.5, -0.5], rtol=1e-6)


def test_denorm_unmasked_dim_passes_through():
    stats = _stats()
    action_norm = np.array([[0.0, 0.0, 0.0, 3.14]], dtype=np.float32)
    out = q99_denorm_with_mask(action_norm, stats)
    assert out[0, 3] == pytest.approx(3.14)


def test_denorm_dim_mismatch_raises():
    stats = _stats(action_dim=4)
    action_norm = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)  # 3 dims, stats want 4
    with pytest.raises(ValueError, match="action_dim"):
        q99_denorm_with_mask(action_norm, stats)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_wire_io_denorm.py -v`
Expected: ModuleNotFoundError or ImportError on `vla_project.deployment.wire_io`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/vla_project/deployment/wire_io.py
"""Wire I/O helpers for the deployment server.

This module replaces the salvageable parts of the old DomainAdapter:
  - q99 denorm with mask (dim-wise q99-inverse on mask=True dims, passthrough on False)
  - JPEG decode + image sanity bounds (Task 2)
  - Proprio normalize + F3 OOD (Task 3)
  - NaN guards (Task 8, called from inference_server)

The contract-translation parts of DomainAdapter (frame conversion,
gripper convention, raw-proprio source/adapt) are NOT recreated here —
those move to clients per the yaml-less spec.
"""
from __future__ import annotations

import numpy as np


def q99_denorm_with_mask(action_norm: np.ndarray, stats: dict) -> np.ndarray:
    """q99-inverse on mask=True dims, passthrough on mask=False dims.

    action_norm: [..., A]
    stats: {"q01": list[A], "q99": list[A], "mask": list[A] bool, ...}
    """
    q01 = np.asarray(stats["q01"], dtype=np.float32)
    q99 = np.asarray(stats["q99"], dtype=np.float32)
    mask = np.asarray(stats["mask"], dtype=bool)
    if action_norm.shape[-1] != q01.shape[0]:
        raise ValueError(
            f"action_dim={action_norm.shape[-1]} != stats dim={q01.shape[0]}"
        )
    span = q99 - q01
    denormed = q01 + (action_norm + 1.0) * 0.5 * span
    return np.where(mask, denormed, action_norm).astype(np.float32)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_wire_io_denorm.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add tests/test_wire_io_denorm.py src/vla_project/deployment/wire_io.py
git commit -m "feat(deploy): wire_io.q99_denorm_with_mask + tests"
```

---

### Task 2: `wire_io.decode_jpeg_b64` + image sanity (F1)

**Files:**
- Test: `tests/test_wire_io_jpeg.py`
- Modify: `src/vla_project/deployment/wire_io.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_wire_io_jpeg.py
"""JPEG decode + F1 image-side sanity (min=64, max=4096)."""
from __future__ import annotations

import base64
import io

import numpy as np
import pytest
from PIL import Image

from vla_project.deployment.wire_io import (
    IMAGE_MAX_SIDE,
    IMAGE_MIN_SIDE,
    decode_jpeg_b64,
)


def _b64_jpeg(h: int, w: int) -> str:
    img = Image.fromarray(np.zeros((h, w, 3), dtype=np.uint8), mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def test_decode_returns_uint8_hwc_rgb():
    arr = decode_jpeg_b64(_b64_jpeg(IMAGE_MIN_SIDE, IMAGE_MIN_SIDE))
    assert arr.dtype == np.uint8
    assert arr.shape == (IMAGE_MIN_SIDE, IMAGE_MIN_SIDE, 3)


def test_decode_min_side_passes():
    decode_jpeg_b64(_b64_jpeg(IMAGE_MIN_SIDE, IMAGE_MIN_SIDE))


def test_decode_max_side_passes():
    decode_jpeg_b64(_b64_jpeg(IMAGE_MAX_SIDE, IMAGE_MAX_SIDE))


def test_decode_below_min_side_rejects():
    with pytest.raises(ValueError, match="below"):
        decode_jpeg_b64(_b64_jpeg(32, 32))


def test_decode_above_max_side_rejects():
    with pytest.raises(ValueError, match="above"):
        decode_jpeg_b64(_b64_jpeg(IMAGE_MAX_SIDE + 1, IMAGE_MAX_SIDE + 1))


def test_decode_lopsided_rejects():
    with pytest.raises(ValueError):
        decode_jpeg_b64(_b64_jpeg(480, 32))  # 32 < min_side
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_wire_io_jpeg.py -v`
Expected: ImportError on `decode_jpeg_b64`, `IMAGE_MIN_SIDE`, `IMAGE_MAX_SIDE`.

- [ ] **Step 3: Append to `wire_io.py`**

```python
# Append to src/vla_project/deployment/wire_io.py:

import base64
import io
from PIL import Image

# F1 image-side sanity bounds. Catches replay corruption (1×1) and
# abusive payloads before JPEG decoder allocates pixel buffers.
IMAGE_MIN_SIDE: int = 64
IMAGE_MAX_SIDE: int = 4096


def decode_jpeg_b64(b64: str) -> np.ndarray:
    """Decode a base64-encoded JPEG into uint8 HWC RGB.

    Raises ValueError if image side is outside [IMAGE_MIN_SIDE, IMAGE_MAX_SIDE]
    on either axis.
    """
    raw = base64.b64decode(b64)
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    h, w = img.height, img.width
    if h < IMAGE_MIN_SIDE or w < IMAGE_MIN_SIDE:
        raise ValueError(
            f"image side below IMAGE_MIN_SIDE={IMAGE_MIN_SIDE}: got h={h}, w={w}"
        )
    if h > IMAGE_MAX_SIDE or w > IMAGE_MAX_SIDE:
        raise ValueError(
            f"image side above IMAGE_MAX_SIDE={IMAGE_MAX_SIDE}: got h={h}, w={w}"
        )
    return np.asarray(img, dtype=np.uint8)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_wire_io_jpeg.py -v`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add tests/test_wire_io_jpeg.py src/vla_project/deployment/wire_io.py
git commit -m "feat(deploy): wire_io.decode_jpeg_b64 + F1 image-side sanity"
```

---

### Task 3: `wire_io.normalize_proprio_q99` + F3 OOD

**Files:**
- Test: `tests/test_wire_io_proprio_ood.py`
- Modify: `src/vla_project/deployment/wire_io.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_wire_io_proprio_ood.py
"""F3 proprio sanity: isfinite + |normed| thresholds after q99 normalize.

Catches: NaN/Inf in proprio (F3a), and degenerate cases like deg/rad
swap which manifest as |normed| → 30+ after q99 (F3b).
"""
from __future__ import annotations

import numpy as np
import pytest

from vla_project.deployment.wire_io import (
    PROPRIO_OOD_HARD_ABS,
    PROPRIO_OOD_WARN_ABS,
    normalize_proprio_q99,
)


def _stats(dim: int = 4) -> dict:
    return {
        "q01":  [-1.0] * dim,
        "q99":  [+1.0] * dim,
        "mask": [True, True, True, False],
        "mean": [0.0] * dim,
        "std":  [1.0] * dim,
        "min":  [-1.0] * dim,
        "max":  [+1.0] * dim,
    }


def test_normalize_within_range_no_clip():
    raw = np.array([0.0, 0.5, -0.5, 0.0], dtype=np.float32)
    out, warned = normalize_proprio_q99(raw, _stats())
    assert warned is False
    np.testing.assert_allclose(out[:3], [0.0, 0.5, -0.5], atol=1e-6)


def test_normalize_passthrough_on_unmasked_dim():
    raw = np.array([0.0, 0.0, 0.0, 99.0], dtype=np.float32)
    out, _ = normalize_proprio_q99(raw, _stats())
    assert out[3] == pytest.approx(99.0)


def test_non_finite_raises():
    raw = np.array([0.0, np.nan, 0.0, 0.0], dtype=np.float32)
    with pytest.raises(ValueError, match="non-finite"):
        normalize_proprio_q99(raw, _stats())


def test_hard_threshold_raises():
    # masked dim values that produce |normed| > PROPRIO_OOD_HARD_ABS after q99.
    raw = np.array([100.0, 0.0, 0.0, 0.0], dtype=np.float32)  # → |normed| = 99 > HARD
    with pytest.raises(ValueError, match="hard"):
        normalize_proprio_q99(raw, _stats())


def test_warn_threshold_clips_and_flags():
    # masked dim values that produce |normed| in (WARN, HARD]
    raw = np.array([3.0, 0.0, 0.0, 0.0], dtype=np.float32)  # → |normed| = 2 > WARN, < HARD
    out, warned = normalize_proprio_q99(raw, _stats())
    assert warned is True
    assert -PROPRIO_OOD_WARN_ABS <= out[0] <= PROPRIO_OOD_WARN_ABS  # clipped
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_wire_io_proprio_ood.py -v`
Expected: ImportError.

- [ ] **Step 3: Append to `wire_io.py`**

```python
# Append to src/vla_project/deployment/wire_io.py:

import logging

logger = logging.getLogger("vla_project.deployment.wire_io")

# F3 proprio OOD thresholds. Computed against normalized values
# (after q01/q99 mapping). Values that exceed PROPRIO_OOD_WARN_ABS but
# stay under HARD are clipped + warned. Values above HARD raise.
# Calibration: 10x the q-range catches deg/rad swap; 1x is the soft
# OOD warning for legitimate startup poses.
PROPRIO_OOD_WARN_ABS: float = 1.0
PROPRIO_OOD_HARD_ABS: float = 10.0


def normalize_proprio_q99(
    proprio_raw: np.ndarray, stats: dict
) -> tuple[np.ndarray, bool]:
    """Normalize raw proprio via q99 to [~ -1, +1] with passthrough on mask=False.

    Returns (normalized, warned). `warned` is True if any masked dim's
    |normed| exceeded PROPRIO_OOD_WARN_ABS (and was clipped). Raises
    ValueError if any value is non-finite or any masked dim's |normed|
    exceeds PROPRIO_OOD_HARD_ABS.
    """
    if not np.isfinite(proprio_raw).all():
        bad_dims = np.where(~np.isfinite(proprio_raw))[0].tolist()
        raise ValueError(f"proprio contains non-finite values at dims {bad_dims}")
    q01 = np.asarray(stats["q01"], dtype=np.float32)
    q99 = np.asarray(stats["q99"], dtype=np.float32)
    mask = np.asarray(stats["mask"], dtype=bool)
    span = q99 - q01
    # F3-aware normalize: protect from div-by-zero on degenerate masked dims.
    safe_span = np.where(span > 0, span, 1.0)
    normed = 2.0 * (proprio_raw - q01) / safe_span - 1.0
    # Passthrough on mask=False dims (gripper passthrough at training time).
    normed = np.where(mask, normed, proprio_raw)
    abs_normed = np.abs(normed)
    hard_viol = (abs_normed > PROPRIO_OOD_HARD_ABS) & mask
    if hard_viol.any():
        bad = np.where(hard_viol)[0].tolist()
        raise ValueError(
            f"proprio normalized |x|>{PROPRIO_OOD_HARD_ABS} (hard) at dims {bad}; "
            f"likely deg/rad swap or wrong proprio dim"
        )
    warn_viol = (abs_normed > PROPRIO_OOD_WARN_ABS) & mask
    warned = bool(warn_viol.any())
    if warned:
        bad = np.where(warn_viol)[0].tolist()
        logger.warning(
            f"proprio normalized |x|>{PROPRIO_OOD_WARN_ABS} (warn) at dims {bad}; clipping"
        )
        normed = np.where(
            mask,
            np.clip(normed, -PROPRIO_OOD_WARN_ABS, PROPRIO_OOD_WARN_ABS),
            normed,
        )
    return normed.astype(np.float32), warned
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_wire_io_proprio_ood.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add tests/test_wire_io_proprio_ood.py src/vla_project/deployment/wire_io.py
git commit -m "feat(deploy): wire_io.normalize_proprio_q99 + F3 OOD checks"
```

---

### Task 4: `post_process_loader.py` with trust gating

**Files:**
- Test: `tests/test_post_process_loader.py`
- Create: `src/vla_project/deployment/post_process_loader.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_post_process_loader.py
"""post_process loader: file presence + trust gating cases per spec §6.

Six cases:
(a) local + valid file       → callable
(b) local + no file          → None
(c) local + no apply         → HardFailAssertion
(d) local + ImportError      → HardFailAssertion
(e) HF + valid + flag off    → None (with WARN log)
(f) HF + valid + flag on     → callable
"""
from __future__ import annotations

import textwrap
from pathlib import Path

import numpy as np
import pytest

from vla_project.deployment.post_process_loader import (
    HardFailAssertion,
    load_post_process,
)


def _write_pp(tmp: Path, body: str) -> Path:
    (tmp / "post_process.py").write_text(textwrap.dedent(body))
    return tmp


def _valid_body() -> str:
    return """\
        import numpy as np
        def apply(actions: np.ndarray, meta: dict) -> np.ndarray:
            actions[..., -1] = 0.5
            return actions
    """


def test_local_valid_returns_callable(tmp_path):
    _write_pp(tmp_path, _valid_body())
    fn = load_post_process(tmp_path, is_local=True, trust_checkpoint_code=False)
    assert callable(fn)
    out = fn(np.zeros((2, 3), dtype=np.float32), meta={})
    assert out[0, -1] == 0.5


def test_local_no_file_returns_none(tmp_path):
    fn = load_post_process(tmp_path, is_local=True, trust_checkpoint_code=False)
    assert fn is None


def test_local_missing_apply_raises(tmp_path):
    _write_pp(tmp_path, "x = 1\n")
    with pytest.raises(HardFailAssertion, match="apply"):
        load_post_process(tmp_path, is_local=True, trust_checkpoint_code=False)


def test_local_import_error_raises(tmp_path):
    _write_pp(tmp_path, "import nonexistent_module_xyz\n")
    with pytest.raises(HardFailAssertion):
        load_post_process(tmp_path, is_local=True, trust_checkpoint_code=False)


def test_hf_no_flag_skips_with_warn(tmp_path, caplog):
    _write_pp(tmp_path, _valid_body())
    with caplog.at_level("WARNING"):
        fn = load_post_process(tmp_path, is_local=False, trust_checkpoint_code=False)
    assert fn is None
    assert any("skipped" in rec.message for rec in caplog.records)


def test_hf_with_flag_returns_callable(tmp_path):
    _write_pp(tmp_path, _valid_body())
    fn = load_post_process(tmp_path, is_local=False, trust_checkpoint_code=True)
    assert callable(fn)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_post_process_loader.py -v`
Expected: ImportError.

- [ ] **Step 3: Create `post_process_loader.py`**

```python
# src/vla_project/deployment/post_process_loader.py
"""Loader for <ckpt_dir>/post_process.py — per-checkpoint action post-processing.

Trust model: model.pt loads with weights_only=True (no RCE), so
post_process.py IS a new RCE surface. Local paths load by default with
a WARN log; HF-resolved paths require explicit --trust-checkpoint-code.
See spec §6 'Trust model'.
"""
from __future__ import annotations

import importlib
import logging
import sys
from pathlib import Path
from typing import Callable

logger = logging.getLogger("vla_project.deployment.post_process_loader")


class HardFailAssertion(Exception):
    """Raised when post_process.py is malformed and the server must not start."""


def load_post_process(
    ckpt_dir: Path,
    *,
    is_local: bool,
    trust_checkpoint_code: bool,
) -> Callable | None:
    """Load post_process.apply from <ckpt_dir>/post_process.py.

    Returns the apply callable, or None if the file is absent or the
    HF trust gate is not opened. Raises HardFailAssertion on malformed
    file (ImportError, missing apply, etc.).
    """
    pp_file = Path(ckpt_dir) / "post_process.py"
    if not pp_file.is_file():
        return None
    if not is_local and not trust_checkpoint_code:
        logger.warning(
            f"{pp_file} present but skipped: ckpt was HF-resolved and "
            f"--trust-checkpoint-code was not passed. Actions returned "
            f"WITHOUT post-processing."
        )
        return None
    sys.path.insert(0, str(ckpt_dir))
    try:
        if "post_process" in sys.modules:
            del sys.modules["post_process"]
        try:
            mod = importlib.import_module("post_process")
        except Exception as e:
            raise HardFailAssertion(
                f"failed to import {pp_file}: {type(e).__name__}: {e}"
            ) from e
        fn = getattr(mod, "apply", None)
        if not callable(fn):
            raise HardFailAssertion(
                f"{pp_file} missing callable apply(actions, meta)"
            )
        logger.warning(
            f"loaded executable post_process from ckpt: {pp_file}. "
            f"This file runs with full server privileges."
        )
        return fn
    finally:
        sys.path.pop(0)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_post_process_loader.py -v`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add tests/test_post_process_loader.py src/vla_project/deployment/post_process_loader.py
git commit -m "feat(deploy): post_process_loader with HF trust gate"
```

---

### Task 5: `startup_validation.py`

**Files:**
- Test: `tests/test_startup_validation.py`
- Create: `src/vla_project/deployment/startup_validation.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_startup_validation.py
"""Startup validation: non-contract checks per spec §8.

Verifies:
  1. domain_id ∈ [0, cfg.model.num_domains)
  2. unnorm_key ∈ meta.norm_stats
  3. action_chunk_len: cfg.data ↔ cfg.model agree (where both exist)
  4. norm_stats[key].action dim ↔ cfg.model action dim
  5. norm_stats[key].proprio dim ↔ cfg.model.proprio_dim
  6. q01/q99/mask/std/min/max shapes agree
  7. wrist hard-required derivation (returns the bool, does not assert)
  8. native_action missing → warn but do not raise

This is logic-only; it does not load a model.
"""
from __future__ import annotations

import logging

import pytest

from vla_project.deployment.startup_validation import (
    HardFailAssertion,
    derive_wrist_hard_required,
    resolve_unnorm_key,
    validate_runtime,
)


def _good_meta(unnorm_key: str = "k", action_dim: int = 7, proprio_dim: int = 8) -> dict:
    return {
        "cfg": {
            "data": {"unnorm_key": unnorm_key, "action_chunk_len": 8, "domain_id": 0},
            "model": {"num_domains": 16, "proprio_dim": proprio_dim, "action_chunk_len": 8},
            "language": {"model_name": "google/gemma-4-E2B"},
        },
        "norm_stats": {
            unnorm_key: {
                "action":  {k: [0.0] * action_dim for k in ("q01", "q99", "mean", "std", "min", "max")}
                          | {"mask": [True] * action_dim},
                "proprio": {k: [0.0] * proprio_dim for k in ("q01", "q99", "mean", "std", "min", "max")}
                          | {"mask": [True] * proprio_dim},
            }
        },
        "step": 1000,
        "git_commit": "abc123",
    }


def test_resolve_unnorm_key_single_auto():
    meta = _good_meta()
    assert resolve_unnorm_key(meta, override=None) == "k"


def test_resolve_unnorm_key_multiple_requires_override():
    meta = _good_meta()
    meta["norm_stats"]["other"] = meta["norm_stats"]["k"]
    with pytest.raises(HardFailAssertion, match="--unnorm-key"):
        resolve_unnorm_key(meta, override=None)


def test_resolve_unnorm_key_override_valid():
    meta = _good_meta()
    assert resolve_unnorm_key(meta, override="k") == "k"


def test_resolve_unnorm_key_override_missing():
    meta = _good_meta()
    with pytest.raises(HardFailAssertion, match="not in"):
        resolve_unnorm_key(meta, override="ghost")


def test_validate_runtime_passes_on_good_meta():
    validate_runtime(_good_meta(), unnorm_key="k", domain_id=0, model_action_dim=7)


def test_validate_runtime_domain_id_out_of_range():
    with pytest.raises(HardFailAssertion, match="domain_id"):
        validate_runtime(_good_meta(), unnorm_key="k", domain_id=100, model_action_dim=7)


def test_validate_runtime_action_dim_mismatch():
    with pytest.raises(HardFailAssertion, match="action_dim"):
        validate_runtime(_good_meta(), unnorm_key="k", domain_id=0, model_action_dim=99)


def test_validate_runtime_q99_shape_mismatch():
    meta = _good_meta()
    meta["norm_stats"]["k"]["action"]["q99"] = [0.0] * 5  # wrong len
    with pytest.raises(HardFailAssertion, match="q99"):
        validate_runtime(meta, unnorm_key="k", domain_id=0, model_action_dim=7)


def test_validate_runtime_missing_native_action_warns(caplog):
    with caplog.at_level("WARNING"):
        validate_runtime(_good_meta(), unnorm_key="k", domain_id=0, model_action_dim=7)
    assert any("native_action" in rec.message for rec in caplog.records)


def test_derive_wrist_hard_required_bridge_true():
    meta = _good_meta()
    meta["cfg"]["model"]["use_wrist_bridge"] = True
    assert derive_wrist_hard_required(meta) is True


def test_derive_wrist_hard_required_dropout_zero_in_llm():
    meta = _good_meta()
    meta["cfg"]["model"]["wrist_in_llm"] = True
    meta["cfg"]["model"]["wrist_view_dropout_p"] = 0.0
    assert derive_wrist_hard_required(meta) is True


def test_derive_wrist_hard_required_dropout_nonzero():
    meta = _good_meta()
    meta["cfg"]["model"]["wrist_in_llm"] = True
    meta["cfg"]["model"]["wrist_view_dropout_p"] = 0.5
    assert derive_wrist_hard_required(meta) is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_startup_validation.py -v`
Expected: ImportError.

- [ ] **Step 3: Create `startup_validation.py`**

```python
# src/vla_project/deployment/startup_validation.py
"""Startup-time validation of meta.json against runtime args.

Spec §8 lists the non-contract checks that survived the yaml-less
refactor: domain_id range, unnorm_key in norm_stats, chunk_len/dim
consistency, q01/q99/mask/std/min/max shape agreement, wrist
hard-required derivation, native_action presence (warn only).
"""
from __future__ import annotations

import logging

logger = logging.getLogger("vla_project.deployment.startup_validation")


class HardFailAssertion(Exception):
    """Raised at startup if meta.json ↔ args are inconsistent."""


_STATS_FIELDS = ("q01", "q99", "mask", "mean", "std", "min", "max")


def resolve_unnorm_key(meta: dict, override: str | None) -> str:
    """Pick the unnorm_key for norm_stats lookup.

    Rules:
      - If override is given, must exist in meta.norm_stats.
      - Else if norm_stats has exactly one key, use it.
      - Else fail with HardFailAssertion (require --unnorm-key).
    """
    keys = list(meta["norm_stats"].keys())
    if override is not None:
        if override not in keys:
            raise HardFailAssertion(
                f"--unnorm-key={override!r} not in meta.norm_stats keys {keys}"
            )
        return override
    if len(keys) == 1:
        return keys[0]
    raise HardFailAssertion(
        f"meta.norm_stats has multiple keys {keys}; pass --unnorm-key"
    )


def derive_wrist_hard_required(meta: dict) -> bool:
    """Whether the model architecture requires wrist_image at request time.

    Hard required if any of:
      - use_wrist_bridge True
      - use_scene_wrist_dinov2_llm / wrist_dinov2 True
      - wrist_in_llm True AND wrist_view_dropout_p == 0.0
    """
    m = meta["cfg"].get("model", {})
    bridge_or_dinov2 = bool(
        m.get("use_wrist_bridge", False)
        or m.get("use_scene_wrist_dinov2_llm", False)
        or m.get("wrist_dinov2", False)
    )
    in_llm = bool(m.get("wrist_in_llm", False))
    dropout = float(m.get("wrist_view_dropout_p") or 0.0)
    return bridge_or_dinov2 or (in_llm and dropout == 0.0)


def validate_runtime(
    meta: dict,
    *,
    unnorm_key: str,
    domain_id: int,
    model_action_dim: int,
) -> None:
    """Run all startup checks (§8). Raises HardFailAssertion on first failure."""
    m_model = meta["cfg"].get("model", {})
    m_data = meta["cfg"].get("data", {})

    # (1) domain_id range
    num_domains = int(m_model.get("num_domains", 0))
    if not (0 <= domain_id < num_domains):
        raise HardFailAssertion(
            f"domain_id={domain_id} out of range [0, {num_domains})"
        )

    # (2) unnorm_key in norm_stats — already enforced by resolve_unnorm_key,
    # but defensive double-check
    if unnorm_key not in meta["norm_stats"]:
        raise HardFailAssertion(
            f"unnorm_key={unnorm_key!r} missing from meta.norm_stats"
        )

    # (3) action_chunk_len: cfg.data ↔ cfg.model agree (if model declares)
    data_chunk = m_data.get("action_chunk_len")
    model_chunk = m_model.get("action_chunk_len")
    if data_chunk is not None and model_chunk is not None and data_chunk != model_chunk:
        raise HardFailAssertion(
            f"action_chunk_len mismatch: cfg.data={data_chunk}, cfg.model={model_chunk}"
        )

    # (4) action stats dim ↔ model action dim
    stats = meta["norm_stats"][unnorm_key]
    action_q99 = stats["action"]["q99"]
    if len(action_q99) != model_action_dim:
        raise HardFailAssertion(
            f"norm_stats.action dim={len(action_q99)} != model action_dim={model_action_dim}"
        )

    # (5) proprio stats dim ↔ cfg.model.proprio_dim
    expected_proprio_dim = int(m_model.get("proprio_dim", 0))
    proprio_q99 = stats["proprio"]["q99"]
    if expected_proprio_dim > 0 and len(proprio_q99) != expected_proprio_dim:
        raise HardFailAssertion(
            f"norm_stats.proprio dim={len(proprio_q99)} != cfg.model.proprio_dim={expected_proprio_dim}"
        )

    # (6) all stats fields agree in shape per block
    for block_name in ("action", "proprio"):
        block = stats[block_name]
        ref_len = len(block["q99"])
        for fld in _STATS_FIELDS:
            if fld not in block:
                raise HardFailAssertion(f"norm_stats.{block_name} missing {fld!r}")
            if len(block[fld]) != ref_len:
                raise HardFailAssertion(
                    f"norm_stats.{block_name}.{fld} len={len(block[fld])} != q99 len={ref_len}"
                )

    # (9) native_action absent → WARN only, do not fail
    if "native_action" not in meta:
        logger.warning(
            "native_action metadata absent; clients must know action convention out-of-band"
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_startup_validation.py -v`
Expected: 11 passed.

- [ ] **Step 5: Commit**

```bash
git add tests/test_startup_validation.py src/vla_project/deployment/startup_validation.py
git commit -m "feat(deploy): startup_validation (non-contract checks)"
```

---

### Task 6: `metadata.py`

**Files:**
- Test: `tests/test_metadata_response.py`
- Create: `src/vla_project/deployment/metadata.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_metadata_response.py
"""/metadata response builder — spec §4 schema."""
from __future__ import annotations

from vla_project.deployment.metadata import build_metadata_response


def _meta(with_native: bool = False) -> dict:
    m = {
        "step": 30000,
        "git_commit": "abc123",
        "cfg": {
            "data": {"unnorm_key": "bottle_pick", "action_chunk_len": 8, "domain_id": 13},
            "model": {"num_domains": 16, "proprio_dim": 8},
            "language": {"model_name": "google/gemma-4-E2B"},
        },
        "norm_stats": {
            "bottle_pick": {
                "action":  {"q99": [0.0] * 7,  "mean": [0.0] * 7,  "mask": [True] * 7},
                "proprio": {"q99": [0.0] * 8,  "mean": [0.0] * 8,  "mask": [True] * 8},
            }
        },
    }
    if with_native:
        m["native_action"] = {
            "units": "meter_axisangle_rad",
            "frame": "world",
            "gripper": {"kind": "absolute", "units": "normalized_0_1",
                        "sign": {"closed": 0, "open": 1}},
        }
    return m


def test_metadata_minimum_fields():
    resp = build_metadata_response(
        _meta(), unnorm_key="bottle_pick", domain_id=13,
        has_post_process=False, post_process_path=None,
    )
    assert resp["step"] == 30000
    assert resp["model_name"] == "google/gemma-4-E2B"
    assert resp["git_commit"] == "abc123"
    assert resp["action_dim"] == 7
    assert resp["proprio_dim"] == 8
    assert resp["action_chunk_len"] == 8
    assert resp["domain_id"] == 13
    assert resp["num_domains"] == 16
    assert resp["unnorm_key"] == "bottle_pick"
    assert resp["native_action"] is None
    assert resp["has_post_process"] is False
    assert resp["post_process_module"] is None


def test_metadata_with_native_action():
    resp = build_metadata_response(
        _meta(with_native=True), unnorm_key="bottle_pick", domain_id=13,
        has_post_process=False, post_process_path=None,
    )
    assert resp["native_action"]["frame"] == "world"
    assert resp["native_action"]["gripper"]["kind"] == "absolute"


def test_metadata_post_process_path():
    resp = build_metadata_response(
        _meta(), unnorm_key="bottle_pick", domain_id=13,
        has_post_process=True, post_process_path="/cache/foo/post_process.py",
    )
    assert resp["has_post_process"] is True
    assert resp["post_process_module"] == "/cache/foo/post_process.py"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_metadata_response.py -v`
Expected: ImportError.

- [ ] **Step 3: Create `metadata.py`**

```python
# src/vla_project/deployment/metadata.py
"""Builder for the GET /metadata response. Spec §4."""
from __future__ import annotations


def build_metadata_response(
    meta: dict,
    *,
    unnorm_key: str,
    domain_id: int,
    has_post_process: bool,
    post_process_path: str | None,
) -> dict:
    cfg = meta["cfg"]
    stats = meta["norm_stats"][unnorm_key]
    return {
        "step": int(meta["step"]),
        "model_name": cfg["language"]["model_name"],
        "git_commit": meta.get("git_commit", ""),
        "action_dim": len(stats["action"]["q99"]),
        "proprio_dim": len(stats["proprio"]["q99"]),
        "action_chunk_len": int(cfg["data"]["action_chunk_len"]),
        "domain_id": int(domain_id),
        "num_domains": int(cfg["model"].get("num_domains", 0)),
        "unnorm_key": unnorm_key,
        "native_action": meta.get("native_action"),
        "has_post_process": bool(has_post_process),
        "post_process_module": post_process_path,
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_metadata_response.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add tests/test_metadata_response.py src/vla_project/deployment/metadata.py
git commit -m "feat(deploy): metadata.build_metadata_response"
```

---

## Phase B — Runtime + server wiring

### Task 7: `runtime.py` — track `is_local` and load `post_process_fn`

**Files:**
- Test: `tests/test_runtime_post_process.py`
- Modify: `src/vla_project/deployment/runtime.py:48-95` (_resolve_ckpt_dir signature change), `:98-198` (ModelRuntime.from_export signature + init)

- [ ] **Step 1: Write the failing test**

```python
# tests/test_runtime_post_process.py
"""ModelRuntime carries is_local and post_process_fn after from_export.

This test does NOT load a real model — it just exercises _resolve_ckpt_dir
and the post_process attachment logic via a local fake ckpt dir.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from vla_project.deployment.runtime import _resolve_ckpt_dir


def test_resolve_local_existing_returns_local_true(tmp_path):
    (tmp_path / "meta.json").write_text("{}")
    resolved, is_local = _resolve_ckpt_dir(tmp_path)
    assert resolved == tmp_path
    assert is_local is True


def test_resolve_absolute_missing_raises(tmp_path):
    ghost = tmp_path / "no_such_dir"
    with pytest.raises(FileNotFoundError):
        _resolve_ckpt_dir(ghost)


def test_resolve_relative_with_dotdot_rejected():
    with pytest.raises(FileNotFoundError):
        _resolve_ckpt_dir("foo/../bar")


# Note: HF-resolution tests are NOT included here — they require a live
# HF call. The is_local=False branch is covered indirectly in the
# inference_server integration test (Task 8) when bottle ckpt is HF-pulled.
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_runtime_post_process.py -v`
Expected: TypeError ("too many values to unpack" — `_resolve_ckpt_dir` returns Path, not tuple).

- [ ] **Step 3: Modify `runtime.py:48-95`**

Replace `_resolve_ckpt_dir` to return `(Path, is_local: bool)`:

```python
def _resolve_ckpt_dir(ckpt_dir: str | Path) -> tuple[Path, bool]:
    """Resolve ckpt_dir → (local directory, is_local).

    is_local=True if the input was a path on disk (we did not call HF).
    is_local=False if we triggered snapshot_download.

    Accepts the same inputs as before (local path, org/repo, org/repo/subfolder).
    """
    p = Path(ckpt_dir)
    if p.exists():
        return p, True
    s = str(ckpt_dir)
    if s.startswith("/") or s.startswith(".") or ".." in s.split("/"):
        raise FileNotFoundError(f"ckpt_dir {ckpt_dir!r} not found locally")
    parts = s.split("/")
    if len(parts) == 2:
        local = Path(snapshot_download(repo_id=str(ckpt_dir), repo_type="model"))
        return local, False
    if len(parts) == 3:
        repo_id = "/".join(parts[:2])
        subfolder = parts[2]
        local = Path(snapshot_download(
            repo_id=repo_id, repo_type="model",
            allow_patterns=[f"{subfolder}/*"],
        ))
        sub = local / subfolder
        if not sub.is_dir():
            raise FileNotFoundError(
                f"resolved HF repo {repo_id!r} but subfolder {subfolder!r} not present in download"
            )
        return sub, False
    raise FileNotFoundError(
        f"ckpt_dir {ckpt_dir!r} not found locally and not in 'org/repo' or "
        f"'org/repo/subfolder' HF form (got {len(parts)} path components)"
    )
```

- [ ] **Step 4: Update `ModelRuntime.from_export` to consume the new tuple**

Modify `runtime.py:123-198`. At line 134, change `ckpt_dir = _resolve_ckpt_dir(ckpt_dir)` to:

```python
ckpt_dir, is_local = _resolve_ckpt_dir(ckpt_dir)
```

Add `is_local` and `post_process_fn` to the `__init__` parameters and `cls(...)` call at line 181. Add `trust_checkpoint_code: bool = False` to `from_export` signature.

After the model is loaded (after line 169) and before the `runtime = cls(...)` block at line 181, insert:

```python
from vla_project.deployment.post_process_loader import load_post_process
post_process_fn = load_post_process(
    ckpt_dir, is_local=is_local, trust_checkpoint_code=trust_checkpoint_code,
)
```

Extend `cls(...)` (line 181) with `is_local=is_local, post_process_fn=post_process_fn, post_process_path=str(ckpt_dir / "post_process.py") if post_process_fn is not None else None`.

Update `__init__` (line 98-120) to accept and store these three new fields.

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/test_runtime_post_process.py -v`
Expected: 3 passed.

- [ ] **Step 6: Run full deployment test subset to ensure nothing regressed**

Run: `uv run pytest tests/test_runtime_load.py tests/test_predictor_xvla_adapter.py -v`
Expected: same pass count as before this task (verify locally; if `test_runtime_load.py` fails because it asserts the old `_resolve_ckpt_dir` return type, update it inline).

- [ ] **Step 7: Commit**

```bash
git add tests/test_runtime_post_process.py src/vla_project/deployment/runtime.py
git commit -m "feat(deploy): runtime tracks is_local + loads post_process_fn"
```

---

### Task 8: Rewrite `inference_server.build_app` — yaml-less

**Files:**
- Test: `tests/test_inference_server_yamlless.py`
- Rewrite: `src/vla_project/deployment/inference_server.py` (full file)

- [ ] **Step 1: Write the failing test**

```python
# tests/test_inference_server_yamlless.py
"""Integration: build_app accepts checkpoint only (no deploy yaml).

Uses a tiny local fake ckpt dir to avoid HF round-trips. The bottle
HF ckpt end-to-end check is a manual smoke (Task 14).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

# Import here to ensure module is importable; build_app is the SUT
from vla_project.deployment.inference_server import build_app


def _fake_local_ckpt(tmp_path: Path) -> Path:
    """Skeleton ckpt with meta.json only — model.pt absent.

    build_app with predictor_kind='hold_position' should NOT need the model.
    """
    meta = {
        "step": 1,
        "git_commit": "test",
        "cfg": {
            "data": {"unnorm_key": "k", "action_chunk_len": 8, "domain_id": 0},
            "model": {"num_domains": 4, "proprio_dim": 8, "action_chunk_len": 8},
            "language": {"model_name": "google/gemma-4-E2B"},
        },
        "norm_stats": {
            "k": {
                "action":  {fld: [0.0] * 7 for fld in ("q01", "q99", "mean", "std", "min", "max")}
                          | {"mask": [True] * 7},
                "proprio": {fld: [0.0] * 8 for fld in ("q01", "q99", "mean", "std", "min", "max")}
                          | {"mask": [True] * 8},
            }
        },
    }
    (tmp_path / "meta.json").write_text(json.dumps(meta))
    return tmp_path


def test_build_app_hold_position_metadata_endpoint(tmp_path):
    ckpt = _fake_local_ckpt(tmp_path)
    app = build_app(
        checkpoint=str(ckpt),
        predictor_kind="hold_position",
        domain_id=0,
        unnorm_key=None,
        trust_checkpoint_code=False,
    )
    client = TestClient(app)
    r = client.get("/metadata")
    assert r.status_code == 200
    body = r.json()
    assert body["step"] == 1
    assert body["action_dim"] == 7
    assert body["proprio_dim"] == 8
    assert body["has_post_process"] is False
    assert body["native_action"] is None


def test_build_app_healthz(tmp_path):
    ckpt = _fake_local_ckpt(tmp_path)
    app = build_app(
        checkpoint=str(ckpt), predictor_kind="hold_position", domain_id=0,
        unnorm_key=None, trust_checkpoint_code=False,
    )
    client = TestClient(app)
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_build_app_rejects_unknown_unnorm_key(tmp_path):
    ckpt = _fake_local_ckpt(tmp_path)
    with pytest.raises(Exception, match="not in"):
        build_app(
            checkpoint=str(ckpt), predictor_kind="hold_position", domain_id=0,
            unnorm_key="ghost", trust_checkpoint_code=False,
        )


def test_build_app_rejects_out_of_range_domain_id(tmp_path):
    ckpt = _fake_local_ckpt(tmp_path)
    with pytest.raises(Exception, match="domain_id"):
        build_app(
            checkpoint=str(ckpt), predictor_kind="hold_position", domain_id=99,
            unnorm_key=None, trust_checkpoint_code=False,
        )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_inference_server_yamlless.py -v`
Expected: TypeError ("got unexpected keyword argument 'unnorm_key'") or similar — current `build_app` does not take these args.

- [ ] **Step 3: Rewrite `inference_server.py` end-to-end**

Replace the entire file. Key changes from current:
- `build_app` signature drops `deploy_config_path`, gains `unnorm_key: str | None`, `trust_checkpoint_code: bool`. Default `predictor_kind="xvla_adapter"`.
- `ModelRuntime.from_export` always loaded (even for `hold_position` — needed for meta.json + norm_stats). If `hold_position`, the model itself can be a fake; treat that as a future smoke-mode concern (note: today, ModelRuntime.from_export will try to load model.pt — we accept this and document that `hold_position` mode still requires meta.json + model.pt files on disk; pure-meta smoke is out of scope).
- All validation goes through `startup_validation.validate_runtime` (no more DomainAdapter).
- The predictor is constructed from runtime + (resolved unnorm_key) only.
- Wire `wire_io.decode_jpeg_b64` + `normalize_proprio_q99` into the `/predict` body, replacing `adapter.preprocess`.
- Wire `runtime.post_process_fn` into the `/predict` body, applied AFTER predictor returns.
- Add `/metadata` route via `metadata.build_metadata_response`.
- Remove `/admin/schema`.

Provide full replacement file content; see [`inference_server.py.template`](#inference-server-template) below the plan.

```python
# src/vla_project/deployment/inference_server.py — full replacement
# (see template appended at the end of this plan for the complete body)
```

(The full template is provided in the appendix to keep this task readable.)

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_inference_server_yamlless.py -v`
Expected: 4 passed.

- [ ] **Step 5: Run full pytest excluding known-to-be-deleted tests**

Run: `uv run pytest tests/ -v --ignore=tests/test_domain_adapter.py --ignore=tests/test_validation_image_sanity.py --ignore=tests/test_validation_proprio.py --ignore=tests/test_admin_schema.py --ignore=tests/test_serve_smoke.py --ignore=tests/test_inference_server_minimal.py`
Expected: All other tests pass. If something else breaks (e.g., an unrelated test that imported DomainAdapter for no reason), fix or note and add to delete list.

- [ ] **Step 6: Commit**

```bash
git add tests/test_inference_server_yamlless.py src/vla_project/deployment/inference_server.py
git commit -m "feat(deploy): rewrite inference_server — yamlless, /metadata route"
```

---

### Task 9: Rewrite `scripts/serve.py` CLI

**Files:**
- Modify: `scripts/serve.py` (full file)

- [ ] **Step 1: Replace the file**

```python
# scripts/serve.py
"""Entry point for the inference HTTP server (yaml-less).

Run with a Hugging Face checkpoint id:
  uv run python scripts/serve.py \\
    --checkpoint takaki99/GEM-4-FT-bottle \\
    --port 8001

Or a local checkpoint dir:
  uv run python scripts/serve.py \\
    --checkpoint outputs/run/checkpoints/step_2000 \\
    --port 8001

See docs/superpowers/specs/2026-05-16-yamlless-hf-deploy-design.md.
"""
from __future__ import annotations

import argparse
import logging
import sys

import uvicorn

from vla_project.deployment.inference_server import build_app


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="X-VLA-Adapter inference HTTP server")
    ap.add_argument("--checkpoint", required=True,
                    help="local ckpt dir, HF id 'org/repo', or 'org/repo/subfolder'")
    ap.add_argument("--predictor", choices=["hold_position", "xvla_adapter"],
                    default="xvla_adapter")
    ap.add_argument("--domain-id", type=int, default=None,
                    help="defaults to cfg.data.domain_id from meta.json")
    ap.add_argument("--unnorm-key", default=None,
                    help="required iff meta.norm_stats has >1 keys")
    ap.add_argument("--trust-checkpoint-code", action="store_true",
                    help="opt-in to load post_process.py from HF-resolved ckpts")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp32"])
    ap.add_argument("--torch-compile", default="off",
                    choices=["off", "reduce-overhead", "default"])
    ap.add_argument("--warmup-iters", type=int, default=1)
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        app = build_app(
            checkpoint=args.checkpoint,
            predictor_kind=args.predictor,
            domain_id=args.domain_id,
            unnorm_key=args.unnorm_key,
            trust_checkpoint_code=args.trust_checkpoint_code,
            device=args.device,
            dtype=args.dtype,
            torch_compile=args.torch_compile,
            warmup_iters=args.warmup_iters,
        )
    except ValueError as e:
        ap.error(str(e))
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Smoke the CLI parser**

Run: `uv run python scripts/serve.py --help`
Expected: argparse help text with `--checkpoint` required, `--deploy-config` ABSENT.

- [ ] **Step 3: Commit**

```bash
git add scripts/serve.py
git commit -m "feat(deploy): scripts/serve.py — new yamlless CLI shape"
```

---

## Phase C — Training side + backfill tool

### Task 10: `training/checkpoint.py` writes `meta.native_action`

**Files:**
- Test: `tests/test_checkpoint_native_action.py`
- Modify: `src/vla_project/training/checkpoint.py`

- [ ] **Step 1: Identify the meta.json writer location**

Run: `grep -n "meta.json\|native_action\|save_meta\|def save_checkpoint" src/vla_project/training/checkpoint.py | head -20`

Confirm there is a function or block that constructs the `meta` dict before writing. (If unclear, read the file to find it.)

- [ ] **Step 2: Write the failing test**

```python
# tests/test_checkpoint_native_action.py
"""checkpoint.save writes meta.native_action when cfg.data.native_action is set."""
from __future__ import annotations

import json
from pathlib import Path

from vla_project.training.checkpoint import build_meta_dict


def test_build_meta_dict_includes_native_action_when_present():
    cfg = {
        "data": {
            "unnorm_key": "x",
            "native_action": {
                "units": "meter_axisangle_rad",
                "frame": "world",
                "gripper": {
                    "kind": "absolute",
                    "units": "normalized_0_1",
                    "sign": {"closed": 0, "open": 1},
                },
            },
        },
        "model": {},
    }
    out = build_meta_dict(step=1, cfg=cfg, norm_stats={}, git_commit="x")
    assert "native_action" in out
    assert out["native_action"]["frame"] == "world"


def test_build_meta_dict_omits_native_action_when_absent():
    cfg = {"data": {"unnorm_key": "x"}, "model": {}}
    out = build_meta_dict(step=1, cfg=cfg, norm_stats={}, git_commit="x")
    assert "native_action" not in out
```

- [ ] **Step 3: Run test to verify it fails**

Run: `uv run pytest tests/test_checkpoint_native_action.py -v`
Expected: ImportError on `build_meta_dict` (likely the function name in the actual file is different — `_make_meta`, inline, etc. — adjust import and refactor as needed).

- [ ] **Step 4: Refactor `checkpoint.py` to expose a `build_meta_dict` function**

Open `src/vla_project/training/checkpoint.py`. Find where `meta = {...}` is constructed before `json.dump`. Extract into `def build_meta_dict(step, cfg, norm_stats, git_commit, tokenizer_settings=None) -> dict:`. Add the native_action lookup:

```python
def build_meta_dict(
    *,
    step: int,
    cfg: dict,
    norm_stats: dict,
    git_commit: str,
    tokenizer_settings: dict | None = None,
) -> dict:
    meta = {
        "step": int(step),
        "cfg": cfg,
        "norm_stats": norm_stats,
        "git_commit": git_commit,
    }
    if tokenizer_settings is not None:
        meta["tokenizer_settings"] = tokenizer_settings
    native_action = cfg.get("data", {}).get("native_action")
    if native_action is not None:
        meta["native_action"] = native_action
    return meta
```

Update the call site (the existing meta-building inline block) to use this helper. Keep all other behavior identical.

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/test_checkpoint_native_action.py -v`
Expected: 2 passed.

- [ ] **Step 6: Run training-side tests to verify no regression**

Run: `uv run pytest tests/test_checkpoint.py tests/test_checkpoint_vla_policy.py tests/test_trainer_autosave.py -v`
Expected: same pass count as before this task.

- [ ] **Step 7: Commit**

```bash
git add tests/test_checkpoint_native_action.py src/vla_project/training/checkpoint.py
git commit -m "feat(train): build_meta_dict supports cfg.data.native_action"
```

---

### Task 11: `tools/backfill_meta_native_action.py`

**Files:**
- Test: `tests/test_backfill_native_action.py`
- Create: `tools/backfill_meta_native_action.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_backfill_native_action.py
"""tools.backfill_meta_native_action: rewrite local meta.json idempotently."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

# We import the main function (not via subprocess) so the test stays fast.
from tools.backfill_meta_native_action import backfill_local


def _write_meta(tmp_path: Path, contents: dict) -> Path:
    p = tmp_path / "meta.json"
    p.write_text(json.dumps(contents))
    return p


def test_adds_native_action_block(tmp_path):
    meta_p = _write_meta(tmp_path, {"step": 1, "cfg": {}})
    backfill_local(
        ckpt_dir=tmp_path,
        units="meter_axisangle_rad",
        frame="world",
        gripper_kind="absolute",
        gripper_units="normalized_0_1",
        gripper_closed=0.0,
        gripper_open=1.0,
    )
    m = json.loads(meta_p.read_text())
    assert m["native_action"]["frame"] == "world"
    assert m["native_action"]["gripper"]["sign"] == {"closed": 0.0, "open": 1.0}


def test_idempotent(tmp_path):
    meta_p = _write_meta(tmp_path, {"step": 1, "cfg": {}})
    args = dict(
        units="meter_axisangle_rad", frame="world",
        gripper_kind="absolute", gripper_units="normalized_0_1",
        gripper_closed=0.0, gripper_open=1.0,
    )
    backfill_local(ckpt_dir=tmp_path, **args)
    first = json.loads(meta_p.read_text())
    backfill_local(ckpt_dir=tmp_path, **args)
    second = json.loads(meta_p.read_text())
    assert first == second


def test_rejects_missing_meta_json(tmp_path):
    with pytest.raises(FileNotFoundError):
        backfill_local(
            ckpt_dir=tmp_path,
            units="meter_axisangle_rad", frame="world",
            gripper_kind="absolute", gripper_units="normalized_0_1",
            gripper_closed=0.0, gripper_open=1.0,
        )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_backfill_native_action.py -v`
Expected: ModuleNotFoundError on `tools.backfill_meta_native_action`.

- [ ] **Step 3: Ensure `tools/` is importable**

Run: `ls tools/ 2>&1`. If `tools/__init__.py` does not exist, create it: `touch tools/__init__.py`.

- [ ] **Step 4: Create the tool**

```python
# tools/backfill_meta_native_action.py
"""Add a native_action block to an existing meta.json.

Usage:
  uv run python tools/backfill_meta_native_action.py \\
    --ckpt /path/to/local/ckpt \\
    --units meter_axisangle_rad --frame world \\
    --gripper-kind absolute --gripper-units normalized_0_1 \\
    --gripper-closed 0.0 --gripper-open 1.0

HF push is manual: edit meta.json locally, then huggingface-cli upload
the modified file to the repo.

The tool rewrites meta.json in-place. It is idempotent: running twice
produces the same file.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def backfill_local(
    *,
    ckpt_dir: Path,
    units: str,
    frame: str,
    gripper_kind: str,
    gripper_units: str,
    gripper_closed: float,
    gripper_open: float,
) -> None:
    meta_p = Path(ckpt_dir) / "meta.json"
    if not meta_p.is_file():
        raise FileNotFoundError(f"meta.json not found at {meta_p}")
    meta = json.loads(meta_p.read_text())
    meta["native_action"] = {
        "units": units,
        "frame": frame,
        "gripper": {
            "kind": gripper_kind,
            "units": gripper_units,
            "sign": {"closed": float(gripper_closed), "open": float(gripper_open)},
        },
    }
    meta_p.write_text(json.dumps(meta, indent=2))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, type=Path,
                    help="local ckpt dir containing meta.json")
    ap.add_argument("--units", default="meter_axisangle_rad")
    ap.add_argument("--frame", required=True, choices=["world", "ee_local"])
    ap.add_argument("--gripper-kind", required=True,
                    choices=["absolute", "delta", "binary"])
    ap.add_argument("--gripper-units", required=True,
                    choices=["normalized_0_1", "signed_neg1_pos1", "binary_threshold_0p5"])
    ap.add_argument("--gripper-closed", type=float, required=True)
    ap.add_argument("--gripper-open", type=float, required=True)
    args = ap.parse_args(argv)
    backfill_local(
        ckpt_dir=args.ckpt,
        units=args.units, frame=args.frame,
        gripper_kind=args.gripper_kind, gripper_units=args.gripper_units,
        gripper_closed=args.gripper_closed, gripper_open=args.gripper_open,
    )
    print(f"wrote native_action to {args.ckpt}/meta.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/test_backfill_native_action.py -v`
Expected: 3 passed.

- [ ] **Step 6: Commit**

```bash
git add tests/test_backfill_native_action.py tools/backfill_meta_native_action.py
[ -f tools/__init__.py ] && git add tools/__init__.py
git commit -m "feat(tools): backfill_meta_native_action for existing ckpts"
```

---

## Phase D — Cleanup

### Task 12: Delete superseded tests

**Files:**
- Delete: 6 test files in `tests/`

- [ ] **Step 1: Identify and confirm**

Run: `git grep -l 'DomainAdapter\|DeployConfig\|load_deploy_config' tests/`

Expected list:
- `tests/test_domain_adapter.py` — entirely about DomainAdapter
- `tests/test_validation_image_sanity.py` — calls `DomainAdapter._decode_jpeg_b64` (superseded by `test_wire_io_jpeg.py`)
- `tests/test_validation_proprio.py` — uses `DomainAdapter` (superseded by `test_wire_io_proprio_ood.py`)
- `tests/test_admin_schema.py` — tests removed `/admin/schema` route
- `tests/test_serve_smoke.py` — tests pre-yamlless serve
- `tests/test_inference_server_minimal.py` — tests pre-yamlless build_app

Also inspect `tests/test_validation_prompt.py` — if it imports DomainAdapter, delete; if it tests prompt tokenizer behavior independently, keep.

- [ ] **Step 2: Decide on `test_validation_prompt.py`**

Run: `grep -n 'DomainAdapter\|load_deploy_config' tests/test_validation_prompt.py`

If grep returns nothing → keep the file. If it returns hits → DELETE.

- [ ] **Step 3: Delete identified files**

```bash
git rm tests/test_domain_adapter.py tests/test_validation_image_sanity.py tests/test_validation_proprio.py tests/test_admin_schema.py tests/test_serve_smoke.py tests/test_inference_server_minimal.py
# Plus tests/test_validation_prompt.py only if Step 2 said DELETE
```

- [ ] **Step 4: Run full pytest**

Run: `uv run pytest tests/ -v --ignore-glob='*deployment*'`
Expected: all remaining tests pass.

Then: `uv run pytest tests/test_wire_io_denorm.py tests/test_wire_io_jpeg.py tests/test_wire_io_proprio_ood.py tests/test_post_process_loader.py tests/test_startup_validation.py tests/test_metadata_response.py tests/test_runtime_post_process.py tests/test_inference_server_yamlless.py tests/test_checkpoint_native_action.py tests/test_backfill_native_action.py -v`
Expected: all new tests pass.

- [ ] **Step 5: Commit**

```bash
git commit -m "test: delete tests superseded by yamlless refactor"
```

---

### Task 13: Delete `domain_adapter.py` + deploy yamls

**Files:**
- Delete: `src/vla_project/deployment/domain_adapter.py`
- Delete: `configs/deploy/*.yaml`

- [ ] **Step 1: Final import scan**

Run: `git grep -l 'domain_adapter\|DomainAdapter\|DeployConfig\|load_deploy_config' src/ scripts/ tools/ tests/`

Expected output: nothing.

If anything turns up → fix or delete it first.

- [ ] **Step 2: Delete the files**

```bash
git rm src/vla_project/deployment/domain_adapter.py
git rm configs/deploy/_template.yaml \
       configs/deploy/so101_v46.yaml \
       configs/deploy/v36_libero_spatial.yaml \
       configs/deploy/mimicrec_pairing_example.yaml
rmdir configs/deploy 2>/dev/null  # ignore failure if other files remain
```

- [ ] **Step 3: Run full pytest**

Run: `uv run pytest tests/ -v`
Expected: full suite green.

- [ ] **Step 4: Commit**

```bash
git commit -m "refactor(deploy): delete domain_adapter.py + configs/deploy/*.yaml"
```

---

### Task 14: End-to-end smoke against bottle HF ckpt

**Files:**
- No new files (manual smoke)

- [ ] **Step 1: Start server**

```bash
CUDA_VISIBLE_DEVICES=0 \
  uv run python scripts/serve.py \
    --checkpoint takaki99/GEM-4-FT-bottle \
    --port 8001 &
sleep 30   # wait for warmup
```

Expected: startup logs include `WARNING: native_action metadata absent` (the bottle ckpt has not been backfilled yet) and either no post_process line (if no flag passed) or `... skipped: ckpt was HF-resolved and --trust-checkpoint-code was not passed`.

- [ ] **Step 2: Hit `/metadata`**

```bash
curl -s http://127.0.0.1:8001/metadata | python -m json.tool
```

Expected fields:
```json
{
  "step": 30000,
  "model_name": "google/gemma-4-E2B",
  "action_dim": 7,
  "proprio_dim": 8,
  "action_chunk_len": 8,
  "domain_id": 13,
  "native_action": null,
  "has_post_process": false
}
```

- [ ] **Step 3: Send a synthetic `/predict` request**

```bash
python <<'PY'
import base64, io, json, requests, numpy as np
from PIL import Image

def b64_jpeg(h=224, w=224):
    img = Image.fromarray(np.zeros((h, w, 3), dtype=np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode("ascii")

req = {
    "image_primary": b64_jpeg(),
    "image_wrist": b64_jpeg(),
    "proprio": [0.0] * 8,
    "instruction": "pick up the bottle",
}
r = requests.post("http://127.0.0.1:8001/predict", json=req, timeout=10)
print(r.status_code, r.json() if r.ok else r.text)
assert r.ok, r.text
chunk = np.array(r.json()["actions"])
print("chunk shape:", chunk.shape)
print("gripper col (dim 6) first 3 rows:", chunk[:3, 6])
PY
```

Expected: `200`, `chunk shape: (8, 7)`. Gripper column (dim 6) is the raw `gripper_pos / 100` passthrough (NOT in `[0, 1]` — because `--trust-checkpoint-code` was not passed; the bottle's post_process is intentionally skipped).

- [ ] **Step 4: Re-run with `--trust-checkpoint-code` (after bottle HF gets a `post_process.py` push)**

This step is OPTIONAL for the PR — requires uploading a `post_process.py` shim to the bottle HF repo manually (using `huggingface-cli upload`). See spec §10 migration note.

If uploaded and re-run: gripper column should now be in `[0, 1]`. If not uploaded yet, skip this step and add a follow-up issue.

- [ ] **Step 5: Kill the server**

```bash
pkill -f 'scripts/serve.py' || true
```

- [ ] **Step 6: Commit (no code, just confirm by going to next task)**

Nothing to commit. Move on.

---

### Task 15: README + docs update

**Files:**
- Modify: `README.md`
- Modify: `src/vla_project/deployment/README.md` (if it exists)

- [ ] **Step 1: Find launch-command sections**

Run: `grep -n 'deploy-config\|--predictor\|configs/deploy' README.md src/vla_project/deployment/README.md 2>/dev/null | head -30`

- [ ] **Step 2: Update `README.md` launch examples**

Replace `--predictor xvla_adapter --checkpoint ... --deploy-config ...` blocks with the new shape:

```bash
# Load directly from Hugging Face Hub.
CUDA_VISIBLE_DEVICES=0 \
  uv run python scripts/serve.py \
    --checkpoint takaki99/GEM-4-FT-bottle \
    --port 8001

# To enable the ckpt-bundled post_process.py for HF-resolved ckpts,
# explicitly opt in:
CUDA_VISIBLE_DEVICES=0 \
  uv run python scripts/serve.py \
    --checkpoint takaki99/GEM-4-FT-bottle \
    --trust-checkpoint-code \
    --port 8001

# Local ckpt directory.
CUDA_VISIBLE_DEVICES=0 \
  uv run python scripts/serve.py \
    --checkpoint outputs/so101_v46_step30k_ft_dl50/checkpoints/step_2000 \
    --port 8001
```

Remove references to `configs/deploy/*.yaml`. Add a one-line note:

> Contract translation (frame conversion, gripper convention mapping, raw proprio adaptation) is the client's responsibility. The server returns model-native, q99-denormalized action chunks. See [`docs/superpowers/specs/2026-05-16-yamlless-hf-deploy-design.md`](docs/superpowers/specs/2026-05-16-yamlless-hf-deploy-design.md).

- [ ] **Step 3: Update available HF ckpts list (if present)**

If README has a "Available HF ckpts" section, leave entries but add `(needs --trust-checkpoint-code for bottle gripper post-process)` next to the bottle entry once that line exists; remove obsolete deploy-yaml references.

- [ ] **Step 4: Same for `src/vla_project/deployment/README.md` if it exists**

Apply the same rewrite.

- [ ] **Step 5: Commit**

```bash
git add README.md
[ -f src/vla_project/deployment/README.md ] && git add src/vla_project/deployment/README.md
git commit -m "docs: README update for yamlless deploy"
```

---

## Codex review checkpoints

Per project CLAUDE.md "Code Review Workflow":

- Before each commit during this implementation: optional, at discretion (TDD with frequent commits → don't bother codex on every step).
- After Task 8 (server rewrite) completes: run `codex review --uncommitted` BEFORE the Task 8 commit — this is the largest semantic change in the plan.
- After Task 15 (entire feature done): run `codex review --base main` BEFORE opening the PR.

If codex flags something, follow the project CLAUDE.md "Per-Round Codex Audit" procedure: write a self-contained re-verification prompt and surface ✅/△/❌ judgments to the user.

---

## Appendix: `inference_server.py` full replacement template

Use this body verbatim in Task 8 Step 3. Length kept tight by removing the F5 admin_schema route and the request-validation handler shim (the handler is moved to be inline within /predict).

```python
"""Yamlless inference HTTP server.

See docs/superpowers/specs/2026-05-16-yamlless-hf-deploy-design.md.

Wires:
  scripts/serve.py CLI args  →  build_app  →  ModelRuntime + new modules
       │
       ├─ runtime.from_export (HF-or-local resolve, model load, post_process_fn)
       ├─ startup_validation.validate_runtime
       ├─ predictor (xvla_adapter | hold_position)
       └─ /healthz, /metadata, /predict routes
"""
from __future__ import annotations

import logging
import time
import uuid
from pathlib import Path
from typing import Any, Literal

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from vla_project.deployment.metadata import build_metadata_response
from vla_project.deployment.predictors.base import ChunkPredictor
from vla_project.deployment.predictors.hold_position import HoldPositionChunkPredictor
from vla_project.deployment.predictors.xvla_adapter import XVLAAdapterChunkPredictor
from vla_project.deployment.runtime import ModelRuntime
from vla_project.deployment.schemas import PredictRequest, PredictResponse
from vla_project.deployment.startup_validation import (
    HardFailAssertion,
    derive_wrist_hard_required,
    resolve_unnorm_key,
    validate_runtime,
)
from vla_project.deployment.wire_io import (
    IMAGE_MAX_SIDE,
    IMAGE_MIN_SIDE,
    PROPRIO_OOD_HARD_ABS,
    PROPRIO_OOD_WARN_ABS,
    decode_jpeg_b64,
    normalize_proprio_q99,
    q99_denorm_with_mask,
)

logger = logging.getLogger("vla_project.deployment")


def build_app(
    *,
    checkpoint: str | Path,
    predictor_kind: Literal["xvla_adapter", "hold_position"] = "xvla_adapter",
    domain_id: int | None = None,
    unnorm_key: str | None = None,
    trust_checkpoint_code: bool = False,
    device: str = "cuda:0",
    dtype: str = "bf16",
    torch_compile: str = "off",
    warmup_iters: int = 1,
    inject_sleep_s: float = 0.0,
) -> FastAPI:
    runtime = ModelRuntime.from_export(
        checkpoint, device=device, dtype=dtype,
        torch_compile=torch_compile, warmup_iters=warmup_iters,
        trust_checkpoint_code=trust_checkpoint_code,
    )
    meta = {
        "step": runtime.step,
        "cfg": runtime.cfg,
        "norm_stats": runtime.norm_stats,
        # native_action / git_commit come from meta.json directly, surfaced
        # through ModelRuntime.meta_raw if the loader exposes it. For now we
        # pass minimal fields; metadata route reads runtime's stored copy.
    }
    # ModelRuntime.from_export already json.loads(meta.json); expose the raw
    # dict via runtime.meta_raw (set in from_export at refactor time).
    meta = runtime.meta_raw if hasattr(runtime, "meta_raw") else meta

    resolved_unnorm_key = resolve_unnorm_key(meta, override=unnorm_key)
    if domain_id is None:
        domain_id = int(meta["cfg"]["data"]["domain_id"])

    # Action dim from the resolved stats; used by validation + predictor.
    action_dim = len(meta["norm_stats"][resolved_unnorm_key]["action"]["q99"])

    validate_runtime(
        meta,
        unnorm_key=resolved_unnorm_key,
        domain_id=domain_id,
        model_action_dim=action_dim,
    )
    wrist_hard_required = derive_wrist_hard_required(meta)

    # Build predictor.
    action_chunk_len = int(meta["cfg"]["data"]["action_chunk_len"])
    if predictor_kind == "hold_position":
        # hold_position does not need the model itself; gripper midpoint
        # defaults to 0.5 (legacy DeployConfig._HoldPosition default).
        predictor: ChunkPredictor = HoldPositionChunkPredictor(
            chunk_len=action_chunk_len,
            action_dim=action_dim,
            gripper_native_midpoint=0.5,
        )
    else:
        predictor = XVLAAdapterChunkPredictor(
            runtime=runtime,
            tokenizer=runtime.tokenizer,
            image_transform=runtime.image_transform,
            action_q99=meta["norm_stats"][resolved_unnorm_key]["action"],
            action_chunk_len=action_chunk_len,
            action_dim=action_dim,
            domain_id=domain_id,
        )

    post_process_fn = runtime.post_process_fn
    post_process_path = runtime.post_process_path
    action_stats = meta["norm_stats"][resolved_unnorm_key]["action"]
    proprio_stats = meta["norm_stats"][resolved_unnorm_key]["proprio"]

    app = FastAPI(title="X-VLA-Adapter Inference Server")
    state = {
        "predictor_class": type(predictor).__name__,
        "ready_at_ns": time.monotonic_ns(),
        "inject_sleep_s": float(inject_sleep_s),
    }

    @app.get("/healthz")
    async def healthz() -> dict:
        return {
            "status": "ok",
            "predictor": state["predictor_class"],
            "ready_at_ns": state["ready_at_ns"],
        }

    @app.get("/metadata")
    async def metadata() -> dict:
        return build_metadata_response(
            meta,
            unnorm_key=resolved_unnorm_key,
            domain_id=domain_id,
            has_post_process=post_process_fn is not None,
            post_process_path=post_process_path,
        )

    @app.exception_handler(RequestValidationError)
    async def _on_validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        errors = jsonable_encoder(exc.errors())
        return JSONResponse(status_code=422, content={"detail": errors})

    @app.post("/predict", response_model=PredictResponse)
    async def predict(req: PredictRequest, request: Request) -> PredictResponse:
        request_id = request.headers.get("X-Request-Id") or uuid.uuid4().hex
        t0 = time.monotonic_ns()
        try:
            # Decode + sanity images.
            scene = decode_jpeg_b64(req.image_primary)
            wrist_was_provided = req.image_wrist is not None
            if wrist_was_provided:
                wrist = decode_jpeg_b64(req.image_wrist)
            elif wrist_hard_required:
                raise ValueError(
                    "checkpoint requires wrist_image but request omitted it"
                )
            else:
                wrist = np.zeros((224, 224, 3), dtype=np.uint8)

            # Normalize proprio.
            proprio_raw = np.asarray(req.proprio, dtype=np.float32)
            if len(proprio_raw) != len(proprio_stats["q99"]):
                raise ValueError(
                    f"proprio length {len(proprio_raw)} != expected "
                    f"{len(proprio_stats['q99'])}"
                )
            proprio_norm, _ = normalize_proprio_q99(proprio_raw, proprio_stats)

            obs = {
                "scene_image": scene,
                "wrist_image": wrist,
                "wrist_was_provided": wrist_was_provided,
                "proprio": proprio_norm,
                "language": req.instruction,
            }
        except Exception as e:
            raise HTTPException(status_code=422, detail=str(e)) from e

        if state["inject_sleep_s"] > 0:
            import asyncio
            await asyncio.sleep(state["inject_sleep_s"])

        try:
            native = predictor.predict(obs)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e)) from e

        if np.isnan(native).any():
            raise HTTPException(status_code=500, detail="predictor emitted NaN")

        # post_process if any (XVLAAdapter already q99-denormed inside predict()).
        if post_process_fn is not None:
            try:
                native = post_process_fn(native, meta)
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"post_process: {e}") from e
            if not isinstance(native, np.ndarray) or native.shape[-1] != action_dim:
                raise HTTPException(
                    status_code=500,
                    detail=f"post_process returned bad shape {getattr(native, 'shape', None)}",
                )
            if np.isnan(native).any():
                raise HTTPException(status_code=500, detail="post_process emitted NaN")

        actions = native.astype(np.float32).tolist()
        elapsed_ms = (time.monotonic_ns() - t0) / 1e6
        logger.info(
            f"predict ok request_id={request_id} elapsed_ms={elapsed_ms:.1f}"
        )
        return PredictResponse(actions=actions)

    return app
```

This template assumes `ModelRuntime.from_export` stores:
- `runtime.meta_raw: dict` (the full loaded meta.json, including any native_action block)
- `runtime.is_local: bool`
- `runtime.post_process_fn: Callable | None`
- `runtime.post_process_path: str | None`

These additions to `runtime.py` happen in Task 7.

---

## Self-review checklist

Before claiming this plan is done:

1. **Spec coverage:** Every section of the spec maps to a task —
   - §3 CLI ⇒ Task 9
   - §4 endpoints ⇒ Task 6, 8
   - §5 native_action ⇒ Task 10, 11
   - §6 post_process ⇒ Task 4, 7, 8
   - §7 delete/move/add ⇒ Tasks 1-3, 12, 13
   - §8 startup validation ⇒ Task 5, 8
   - §9 tests ⇒ each Task includes its TDD test
   - §10 migration ⇒ Task 14 (smoke), Task 15 (README note)
   - §11 out of scope ⇒ not implemented (correct)

2. **Placeholders:** None — every step shows the exact code, command, or list.

3. **Type consistency:**
   - `q99_denorm_with_mask(action_norm, stats)` used in Task 1 + Task 8 — consistent.
   - `normalize_proprio_q99(raw, stats) -> (np.ndarray, bool)` used in Task 3 + Task 8 — consistent.
   - `load_post_process(ckpt_dir, *, is_local, trust_checkpoint_code) -> Callable | None` used in Task 4 + Task 7 — consistent.
   - `build_metadata_response(meta, *, unnorm_key, domain_id, has_post_process, post_process_path)` used in Task 6 + Task 8 — consistent.
   - `validate_runtime(meta, *, unnorm_key, domain_id, model_action_dim)` used in Task 5 + Task 8 — consistent.
   - `build_meta_dict(*, step, cfg, norm_stats, git_commit, tokenizer_settings)` used in Task 10 only.
   - `backfill_local(*, ckpt_dir, units, frame, gripper_kind, gripper_units, gripper_closed, gripper_open)` used in Task 11 only.
