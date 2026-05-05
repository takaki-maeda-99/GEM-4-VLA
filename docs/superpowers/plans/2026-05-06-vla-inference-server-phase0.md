# VLA Inference HTTP Server — Phase 0 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver every item in the Phase 0 acceptance gate (spec §"Phase 0 acceptance gate") — a FastAPI server that boots in HoldPosition mode, passes MimicRec's `smoke_inference_real_data.py`, and has 6 named test files green.

**Architecture:** Monolithic FastAPI process. 4-layer stack:
```
[FastAPI route] → [DomainAdapter] → [ChunkPredictor] → [ModelRuntime stub]
```
Phase 0 ships `HoldPositionChunkPredictor` end-to-end. `XVLAAdapterChunkPredictor` is a typed shell that raises `NotImplementedError` on `predict()` — Phase 1 fills in. `ModelRuntime` is similarly a stub that loads `meta.json` for assertion only; full torch forward path is Phase 1.

**Tech Stack:** Python 3.10+, uv, FastAPI, uvicorn, pydantic v2, pytest, numpy, PyYAML.

**Reference spec:** `docs/superpowers/specs/2026-05-06-vla-inference-server-design.md` (committed at `ce313ad`). All design decisions, schemas, validation rules, and acceptance criteria live there. This plan only schedules the implementation.

**Codex review cadence:** per CLAUDE.md "Code Review Workflow" (and global CLAUDE.md "Second-opinion reviews with Codex"), run `codex exec -m gpt-5.5 --skip-git-repo-check '<prompt>'` BEFORE each commit. Each task below has an explicit codex step. Treat findings as a second opinion, not authority.

---

## File Structure

New files (all created by this plan):

```
src/vla_project/deployment/
├── __init__.py                      # exports build_app
├── inference_server.py              # build_app() + FastAPI route + middleware
├── schemas.py                       # PredictRequest, PredictResponse pydantic v2
├── domain_adapter.py                # DomainAdapter, DeployConfig pydantic, validators
├── runtime.py                       # ModelRuntime stub (Phase 1 fills forward path)
├── README.md                        # operator-facing docs
└── predictors/
    ├── __init__.py                  # exports ChunkPredictor + concrete classes
    ├── base.py                      # ChunkPredictor ABC
    ├── hold_position.py             # HoldPositionChunkPredictor (full impl)
    └── xvla_adapter.py              # XVLAAdapterChunkPredictor stub (Phase 1)

scripts/
└── serve.py                         # argparse → uvicorn.run(build_app(...))

configs/deploy/
├── _template.yaml                   # commented schema reference
├── v36_libero_spatial.yaml          # first concrete deploy
└── mimicrec_pairing_example.yaml    # paste-and-edit example for MimicRec side
                                     # (informational; lives outside MimicRec repo on purpose)

tests/
├── test_deployment_schemas.py
├── test_domain_adapter.py
├── test_predictor_holdposition.py
├── test_predictor_xvla_adapter.py
├── test_runtime_load.py
└── test_serve_smoke.py
```

Files modified:
- `pyproject.toml` (add fastapi, uvicorn, httpx[for TestClient] to dependencies)

Files NOT touched in Phase 0:
- `src/vla_project/policies/xvla_adapter_policy.py` (rollout-side policy — stays as-is per spec §"Reuse with existing XVLAAdapterPolicy")
- `src/vla_project/models/*` (no model code changes)

---

## Glossary of canonical names (consistent across tasks)

| Symbol | Definition |
|---|---|
| `PredictRequest` | pydantic v2 model. Fields: `image_primary: str` (base64 JPEG), `image_wrist: str \| None`, `proprio: list[float]`, `instruction: str`, `model_version: str \| None`, `t_mono_ns: dict \| None` (alias `_t_mono_ns`). |
| `PredictResponse` | pydantic v2 model. Field: `actions: list[list[float]]`. |
| `Obs` | dict produced by `DomainAdapter.preprocess`. Keys: `scene_image: np.uint8[H,W,3]`, `wrist_image: np.uint8[H,W,3] \| None`, `wrist_was_provided: bool`, `proprio: np.float32[D_prop]`, `language: str`. |
| `ChunkPredictor` | ABC. `predict(obs: dict) -> np.ndarray[T, A]` in NATIVE units. Properties: `chunk_len: int`, `action_dim: int`. |
| `HoldPositionChunkPredictor(chunk_len, action_dim, gripper_native_midpoint=0.5)` | Phase 0 concrete. Returns zeros except last column = midpoint. |
| `XVLAAdapterChunkPredictor(runtime, tokenizer, image_transform, action_q99, action_chunk_len, action_dim, domain_id)` | Phase 1 concrete; Phase 0 = stub raising `NotImplementedError`. |
| `DomainAdapter(deploy_config, norm_stats, domain_id)` | `preprocess(req: PredictRequest) -> dict`, `postprocess(native_chunk: np.ndarray) -> list[list[float]]`. `norm_stats` may be `None` for HoldPosition mode. |
| `DeployConfig` | pydantic v2 model parsed from `configs/deploy/<robot>_<model>.yaml`. |
| `ModelRuntime.from_export(ckpt_dir, device, dtype, torch_compile, warmup_iters)` | classmethod. Phase 0 stub: only loads `meta.json` for assertion; raises `NotImplementedError` on `__call__`. **Phase 0 dtype is `str` (`"bf16"`/`"fp32"`) carried straight from `deploy.runtime.dtype`; Phase 1 will introduce `torch.dtype` conversion at the model load site (spec line 510 references `torch.dtype` for the Phase 1 signature).** |
| Field-name mapping note | The pydantic schema fields (`image_primary`, `image_wrist`, `proprio`, `instruction`) are the **canonical wire names**. `deploy.request_fields` documents the expected mapping but Phase 0 does not rename arbitrary contract field names — the MimicRec contract YAML for this server MUST use the canonical names. Phase 1 may add pydantic alias remapping to support contracts with different names. |
| `build_app(predictor_kind, checkpoint, deploy_config_path, domain_id, inject_sleep_s) -> FastAPI` | Top-level factory. `predictor_kind ∈ {"hold_position", "xvla_adapter"}`. |

---

## Task 1: Dependencies + package skeleton

**Files:**
- Modify: `pyproject.toml`
- Create: `src/vla_project/deployment/__init__.py`
- Create: `src/vla_project/deployment/predictors/__init__.py`
- Create: `tests/test_deployment_imports.py` (smoke; deleted in Task 2 once real tests exist)

- [ ] **Step 1: Add server deps to pyproject.toml**

Modify `pyproject.toml` `[project] dependencies` (add to the existing list):

```toml
dependencies = [
    # ... existing deps ...
    "fastapi>=0.115",
    "uvicorn[standard]>=0.32",
    "httpx>=0.27",       # for FastAPI TestClient
    "pyyaml>=6.0",       # explicit (currently transitive)
]
```

Run:
```bash
uv sync
```

- [ ] **Step 2: Create empty package + smoke test**

```python
# src/vla_project/deployment/__init__.py
"""HTTP inference server for X-VLA-Adapter checkpoints.

See docs/superpowers/specs/2026-05-06-vla-inference-server-design.md for the
full design. This package implements the Phase 0 HoldPosition path; the
XVLAAdapterChunkPredictor full forward path is Phase 1.
"""
```

```python
# src/vla_project/deployment/predictors/__init__.py
"""ChunkPredictor implementations.

See deployment/predictors/base.py for the ABC contract.
"""
```

```python
# tests/test_deployment_imports.py
"""Smoke: deployment package imports without errors. Will be replaced by
test_deployment_schemas.py in Task 2."""

def test_deployment_package_imports():
    import vla_project.deployment  # noqa: F401
    import vla_project.deployment.predictors  # noqa: F401
```

- [ ] **Step 3: Run smoke test**

```bash
uv run pytest tests/test_deployment_imports.py -v
```

Expected: PASS.

- [ ] **Step 4: codex review**

```bash
codex exec -m gpt-5.5 --skip-git-repo-check 'Review uncommitted diff. New files: src/vla_project/deployment/__init__.py, src/vla_project/deployment/predictors/__init__.py, tests/test_deployment_imports.py, plus pyproject.toml dependency additions (fastapi, uvicorn[standard], httpx, pyyaml). Goal: bootstrap deployment package per docs/superpowers/specs/2026-05-06-vla-inference-server-design.md §Section 2 module boundaries. Concerns: dependency version choices reasonable for FastAPI 0.115 + pydantic v2.13? Any imports/exports missing for downstream tasks? Reply terse, line numbers, no preamble.'
```

Triage findings — apply if correct, push back if not.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml uv.lock src/vla_project/deployment/__init__.py src/vla_project/deployment/predictors/__init__.py tests/test_deployment_imports.py
git commit -m "$(cat <<'EOF'
feat(deployment): bootstrap inference server package skeleton

Adds fastapi, uvicorn[standard], httpx, pyyaml to project deps.
Creates empty deployment/ + deployment/predictors/ packages with module
docstrings pointing at the design spec. Smoke test verifies imports.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Pydantic schemas

**Files:**
- Create: `src/vla_project/deployment/schemas.py`
- Create: `tests/test_deployment_schemas.py`
- Delete: `tests/test_deployment_imports.py` (superseded)

- [ ] **Step 1: Write failing tests**

```python
# tests/test_deployment_schemas.py
"""Tests for deployment.schemas — pydantic v2 PredictRequest / PredictResponse.

Spec §Section 3 "Per-request data flow" defines the wire shape MimicRec sends.
"""
import base64
import pytest
from pydantic import ValidationError

from vla_project.deployment.schemas import PredictRequest, PredictResponse


def _b64_jpeg(n_bytes: int = 64) -> str:
    return base64.b64encode(b"\xff\xd8\xff" + b"\x00" * n_bytes).decode("ascii")


def test_predict_request_minimal_round_trip():
    """Required fields only: image_primary, proprio, instruction."""
    req = PredictRequest(
        image_primary=_b64_jpeg(),
        proprio=[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 0.5],
        instruction="pick up the bottle",
    )
    assert req.image_primary.startswith("/9j/")  # base64 JPEG marker
    assert req.image_wrist is None
    assert req.proprio == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 0.5]
    assert req.instruction == "pick up the bottle"
    assert req.model_version is None
    assert req.t_mono_ns is None


def test_predict_request_full_with_aliased_underscore_field():
    """The wire field is `_t_mono_ns` (underscore-prefixed) — pydantic v2 needs
    populate_by_name + alias to expose it as `t_mono_ns` on the model."""
    req = PredictRequest.model_validate({
        "image_primary": _b64_jpeg(),
        "image_wrist": _b64_jpeg(),
        "proprio": [0.0] * 7,
        "instruction": "stir the pot",
        "model_version": "x_vla_v36",
        "_t_mono_ns": {"state": 1, "image:front": 2},
    })
    assert req.image_wrist is not None
    assert req.model_version == "x_vla_v36"
    assert req.t_mono_ns == {"state": 1, "image:front": 2}


def test_predict_request_missing_image_primary_raises():
    with pytest.raises(ValidationError):
        PredictRequest.model_validate({
            "proprio": [0.0] * 7,
            "instruction": "x",
        })


def test_predict_request_proprio_must_be_list_of_numbers():
    with pytest.raises(ValidationError):
        PredictRequest.model_validate({
            "image_primary": _b64_jpeg(),
            "proprio": "not a list",
            "instruction": "x",
        })


def test_predict_request_instruction_can_be_empty_string():
    """Spec §Section 3 says empty instruction is valid in pre-start states."""
    req = PredictRequest(
        image_primary=_b64_jpeg(),
        proprio=[0.0] * 7,
        instruction="",
    )
    assert req.instruction == ""


def test_predict_response_round_trip():
    resp = PredictResponse(actions=[[0.0] * 7 for _ in range(8)])
    dumped = resp.model_dump()
    assert dumped == {"actions": [[0.0] * 7 for _ in range(8)]}


def test_predict_response_actions_must_be_list_of_lists():
    with pytest.raises(ValidationError):
        PredictResponse.model_validate({"actions": "not a list"})
```

- [ ] **Step 2: Run tests → verify all fail with import / schema errors**

```bash
uv run pytest tests/test_deployment_schemas.py -v
```

Expected: errors like `ImportError: cannot import name 'PredictRequest' from 'vla_project.deployment.schemas'`.

- [ ] **Step 3: Implement schemas**

```python
# src/vla_project/deployment/schemas.py
"""Pydantic v2 wire schemas for the inference HTTP server.

PredictRequest mirrors what MimicRec sends per its contract YAML; PredictResponse
is what the server returns. Field naming follows the MimicRec spec excerpts in
docs/superpowers/specs/2026-05-06-vla-inference-server-design.md §Section 3.

The wire field `_t_mono_ns` is exposed as `t_mono_ns` on the model because
pydantic v2 reserves leading-underscore names as private attributes; we use
`populate_by_name=True` + `Field(alias="_t_mono_ns")`.
"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class PredictRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    image_primary: str
    image_wrist: str | None = None
    proprio: list[float]
    instruction: str
    model_version: str | None = None
    t_mono_ns: dict[str, Any] | None = Field(default=None, alias="_t_mono_ns")


class PredictResponse(BaseModel):
    actions: list[list[float]]
```

- [ ] **Step 4: Delete superseded smoke test**

```bash
rm tests/test_deployment_imports.py
```

- [ ] **Step 5: Run tests → all pass**

```bash
uv run pytest tests/test_deployment_schemas.py -v
```

Expected: all 7 tests PASS.

- [ ] **Step 6: codex review**

```bash
codex exec -m gpt-5.5 --skip-git-repo-check 'Review uncommitted diff. New file src/vla_project/deployment/schemas.py implements PredictRequest / PredictResponse per docs/superpowers/specs/2026-05-06-vla-inference-server-design.md §Section 3. Note: extra="ignore" tolerates contract-side extra_fields like model_version-variants. Concerns: any pydantic v2 idiom that breaks with the underscore-aliased field? Any required field I missed (cross-check spec lines 145-153)? Reply terse, line numbers.'
```

- [ ] **Step 7: Commit**

```bash
git add src/vla_project/deployment/schemas.py tests/test_deployment_schemas.py tests/test_deployment_imports.py
git commit -m "$(cat <<'EOF'
feat(deployment): pydantic PredictRequest / PredictResponse schemas

Wire-format pydantic v2 models matching MimicRec's POST /predict contract.
Aliases `_t_mono_ns` (underscore-prefixed wire field) → `t_mono_ns` model
attribute via populate_by_name + Field alias. Tolerates extra contract-side
fields with extra="ignore".

7 tests cover: minimal request, full request with aliased timing, missing
required field, type-mismatch proprio, empty instruction (pre-start state),
response round-trip, response shape validation.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: ChunkPredictor ABC

**Files:**
- Create: `src/vla_project/deployment/predictors/base.py`
- Create: `tests/test_predictor_base.py` (small ABC contract test; merged into existing test files later if redundant)

- [ ] **Step 1: Write failing tests**

```python
# tests/test_predictor_base.py
"""ABC contract: ChunkPredictor cannot be instantiated, subclasses MUST
override predict / chunk_len / action_dim."""
import numpy as np
import pytest

from vla_project.deployment.predictors.base import ChunkPredictor


def test_chunk_predictor_is_abstract():
    with pytest.raises(TypeError, match="abstract"):
        ChunkPredictor()  # type: ignore[abstract]


def test_concrete_subclass_must_override_predict():
    class Bad(ChunkPredictor):
        @property
        def chunk_len(self) -> int: return 1
        @property
        def action_dim(self) -> int: return 1
    with pytest.raises(TypeError, match="abstract"):
        Bad()  # type: ignore[abstract]


def test_concrete_subclass_with_all_overrides_works():
    class Good(ChunkPredictor):
        def predict(self, obs):
            return np.zeros((1, 1), dtype=np.float32)
        @property
        def chunk_len(self) -> int: return 1
        @property
        def action_dim(self) -> int: return 1
    p = Good()
    assert p.chunk_len == 1
    assert p.action_dim == 1
    assert p.predict({}).shape == (1, 1)
```

- [ ] **Step 2: Run → verify import errors**

```bash
uv run pytest tests/test_predictor_base.py -v
```

Expected: ImportError.

- [ ] **Step 3: Implement ABC**

```python
# src/vla_project/deployment/predictors/base.py
"""ChunkPredictor ABC.

A predictor takes a fully-prepped Obs dict (post DomainAdapter.preprocess)
and returns a (T, A) np.float32 chunk in MODEL NATIVE physical units.
DomainAdapter.postprocess handles frame / gripper-convention conversion to
MimicRec contract units.

For v36 (and v33/v35 RLDS-trained), native gripper is normalized_0_1
(closed=0, open=1), frame is LIBERO world frame, deltas are
meter+axisangle_rad. See docs/superpowers/specs/2026-05-06-vla-inference-server-design.md
§Section 5.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class ChunkPredictor(ABC):
    @abstractmethod
    def predict(self, obs: dict[str, Any]) -> np.ndarray:
        """Return one chunk in NATIVE units, shape (T, A) np.float32."""

    @property
    @abstractmethod
    def chunk_len(self) -> int: ...

    @property
    @abstractmethod
    def action_dim(self) -> int: ...
```

- [ ] **Step 4: Run → all pass**

```bash
uv run pytest tests/test_predictor_base.py -v
```

Expected: 3 PASS.

- [ ] **Step 5: codex review**

```bash
codex exec -m gpt-5.5 --skip-git-repo-check 'Review src/vla_project/deployment/predictors/base.py + tests/test_predictor_base.py. ABC matches spec §Section 5 lines 400-431. Concerns: any signature drift from spec? Should obs typed with TypedDict for clarity? Reply terse.'
```

- [ ] **Step 6: Commit**

```bash
git add src/vla_project/deployment/predictors/base.py tests/test_predictor_base.py
git commit -m "$(cat <<'EOF'
feat(deployment): ChunkPredictor ABC

Abstract base class for chunk predictors. predict(obs) returns (T, A)
np.float32 in MODEL NATIVE units; postprocess converts to contract units
in DomainAdapter (separate concern).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: HoldPositionChunkPredictor

**Files:**
- Create: `src/vla_project/deployment/predictors/hold_position.py`
- Create: `tests/test_predictor_holdposition.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_predictor_holdposition.py
"""Tests for HoldPositionChunkPredictor — emit zero ee_delta + native midpoint
gripper. See spec §Section 5 lines 433-460 for the design rationale (zero
across all columns would silently command CLOSED for normalized_0_1 native)."""
import numpy as np
import pytest

from vla_project.deployment.predictors.hold_position import HoldPositionChunkPredictor


def test_chunk_len_and_action_dim_are_constructor_args():
    p = HoldPositionChunkPredictor(chunk_len=8, action_dim=7)
    assert p.chunk_len == 8
    assert p.action_dim == 7


def test_predict_returns_zeros_for_ee_delta_columns():
    p = HoldPositionChunkPredictor(chunk_len=8, action_dim=7)
    out = p.predict(obs={})
    assert out.shape == (8, 7)
    assert out.dtype == np.float32
    np.testing.assert_array_equal(out[:, :6], np.zeros((8, 6), dtype=np.float32))


def test_predict_gripper_column_is_default_midpoint():
    p = HoldPositionChunkPredictor(chunk_len=8, action_dim=7)
    out = p.predict(obs={})
    np.testing.assert_array_equal(out[:, 6], np.full(8, 0.5, dtype=np.float32))


def test_predict_gripper_column_uses_configured_midpoint():
    p = HoldPositionChunkPredictor(chunk_len=8, action_dim=7, gripper_native_midpoint=0.0)
    out = p.predict(obs={})
    np.testing.assert_array_equal(out[:, 6], np.zeros(8, dtype=np.float32))


def test_predict_obs_is_unused_does_not_raise():
    """HoldPosition does not read obs at all."""
    p = HoldPositionChunkPredictor(chunk_len=4, action_dim=7)
    out_a = p.predict({})
    out_b = p.predict({"scene_image": "garbage"})
    np.testing.assert_array_equal(out_a, out_b)
```

- [ ] **Step 2: Run → verify ImportError**

```bash
uv run pytest tests/test_predictor_holdposition.py -v
```

Expected: ImportError.

- [ ] **Step 3: Implement HoldPosition**

```python
# src/vla_project/deployment/predictors/hold_position.py
"""HoldPositionChunkPredictor — wire-shape smoke / pre-model-trained sentinel.

NOT a production safety fallback (MimicRec's slow-stop ramp is the real
fallback). Emits zero ee_delta for cols 0..5 and `gripper_native_midpoint`
for col 6 (in MODEL NATIVE gripper units; postprocess converts to contract).

For v36 (and v33/v35) native = normalized_0_1 (closed=0, open=1), midpoint
0.5 lands on contract midpoint. For signed_neg1_pos1 native, set 0.0.

See docs/superpowers/specs/2026-05-06-vla-inference-server-design.md
§Section 5 (HoldPositionChunkPredictor).
"""
from __future__ import annotations

from typing import Any

import numpy as np

from vla_project.deployment.predictors.base import ChunkPredictor


class HoldPositionChunkPredictor(ChunkPredictor):
    def __init__(
        self,
        chunk_len: int,
        action_dim: int,
        gripper_native_midpoint: float = 0.5,
    ) -> None:
        self._T = int(chunk_len)
        self._A = int(action_dim)
        self._g = float(gripper_native_midpoint)

    @property
    def chunk_len(self) -> int:
        return self._T

    @property
    def action_dim(self) -> int:
        return self._A

    def predict(self, obs: dict[str, Any]) -> np.ndarray:  # noqa: ARG002 (obs unused)
        a = np.zeros((self._T, self._A), dtype=np.float32)
        a[:, -1] = self._g
        return a
```

- [ ] **Step 4: Run → all pass**

```bash
uv run pytest tests/test_predictor_holdposition.py -v
```

Expected: 5 PASS.

- [ ] **Step 5: codex review**

```bash
codex exec -m gpt-5.5 --skip-git-repo-check 'Review src/vla_project/deployment/predictors/hold_position.py. Implements spec §Section 5 lines 433-460 HoldPositionChunkPredictor: zero ee_delta + native midpoint gripper. 5 tests (constructor args, ee_delta cols zero, default + custom midpoint, obs ignored). Concerns: any case where last column is not the gripper? If action_dim < 1, what happens? (currently silently no-ops via the slice). Reply terse.'
```

- [ ] **Step 6: Commit**

```bash
git add src/vla_project/deployment/predictors/hold_position.py tests/test_predictor_holdposition.py
git commit -m "$(cat <<'EOF'
feat(deployment): HoldPositionChunkPredictor

Wire-shape smoke / pre-model-trained sentinel. Emits zero ee_delta + native
midpoint gripper (default 0.5 for normalized_0_1 native). NOT a production
safety fallback — MimicRec's slow-stop is the real fallback.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: DomainAdapter + DeployConfig

**Files:**
- Create: `src/vla_project/deployment/domain_adapter.py`
- Create: `tests/test_domain_adapter.py`

This is the largest task — it covers (a) DeployConfig pydantic schema parsing, (b) preprocess (JPEG decode + field map + proprio adapt + Q99 normalize), (c) postprocess (gripper conversion + frame conversion = none in Phase 0 + row-shape assert), (d) validation of the assertions from spec §"Startup validation flow".

- [ ] **Step 1: Write failing tests (split into multiple test classes for readability)**

```python
# tests/test_domain_adapter.py
"""Tests for DomainAdapter and DeployConfig.

Spec §Section 4 deploy YAML schema, §Section 3 per-request data flow.
Covered:
- DeployConfig pydantic round-trip + per-field validation
- proprio.adapt step ops (deg_to_rad, copy, pad_zeros)
- Q99 normalize/denormalize with mask handling
- gripper convention conversion (normalized_0_1 ↔ signed_neg1_pos1, sign flip)
- frame_conversion=none identity
- row-shape postprocess assert (input [T, 7] ok; [T, 6] / [T, 8] raises)
- startup hard-fail assertions: domain_id < 0, mismatched unnorm_key, etc.
"""
from __future__ import annotations

import base64
import io

import numpy as np
import pytest
import yaml
from PIL import Image
from pydantic import ValidationError

from vla_project.deployment.domain_adapter import (
    DeployConfig,
    DomainAdapter,
    HardFailAssertion,
    load_deploy_config,
)
from vla_project.deployment.schemas import PredictRequest


# ---------- helpers ----------

def _make_jpeg_b64(size: int = 224) -> str:
    img = Image.new("RGB", (size, size), color=(127, 127, 127))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _minimal_deploy_yaml(**overrides) -> dict:
    """Returns a valid DeployConfig dict matching v36 + SO-101 contract."""
    base = {
        "ckpt": {
            "expected_unnorm_key": "libero_spatial_no_noops",
            "expected_action_chunk_len": 8,
            "expected_action_dim": 7,
            "expected_proprio_dim": 8,
        },
        "request_fields": {
            "scene_image": "image_primary",
            "wrist_image": "image_wrist",
            "proprio": "proprio",
            "instruction": "instruction",
        },
        "proprio": {
            "source": {
                "components": [
                    {"name": "joint_pos", "dims": 6, "units": "deg"},
                    {"name": "gripper_pos", "dims": 1, "units": "normalized_neg1_pos1"},
                ],
                "total_dim": 7,
            },
            "adapt": {
                "steps": [
                    {"op": "deg_to_rad", "source": "joint_pos", "dims": 6},
                    {"op": "copy", "source": "gripper_pos", "dims": 1},
                    {"op": "pad_zeros", "dims": 1},
                ],
                "output_dim": 8,
            },
            "normalization": {"method": "q99", "stats_key": "proprio"},
        },
        "action": {
            "native": {
                "units": "meter_axisangle_rad",
                "frame": "world",
                "gripper": {
                    "kind": "absolute",
                    "units": "normalized_0_1",
                    "sign": {"closed": 0, "open": 1},
                },
            },
            "contract": {
                "units": "meter_axisangle_rad",
                "frame": "ee_local",
                "gripper": {
                    "kind": "absolute",
                    "units": "normalized_0_1",
                    "sign": {"closed": 0, "open": 1},
                },
            },
            "denormalization": {"method": "q99", "stats_key": "action"},
            "frame_conversion": {"method": "none"},
        },
        "holdposition": {"gripper_native_midpoint": 0.5},
        "wire_only_smoke": True,  # set true so v36 world->ee_local mismatch passes startup for tests
        "runtime": {
            "device": "cpu",
            "dtype": "bf16",
            "torch_compile": "off",
            "warmup_iters": 0,
        },
    }
    for k, v in overrides.items():
        base[k] = v
    return base


def _norm_stats_v36() -> dict:
    """Subset of meta.norm_stats[unnorm_key] sufficient for D_prop=8 + A=7."""
    return {
        "action": {
            "mean": [0.0] * 7,
            "std": [1.0] * 7,
            "q01": [-1.0] * 7,
            "q99": [1.0] * 7,
            "mask": [True, True, True, True, True, True, False],  # gripper dim passes through
        },
        "proprio": {
            "mean": [0.0] * 8,
            "std": [1.0] * 8,
            "q01": [-1.0] * 8,
            "q99": [1.0] * 8,
            "mask": [True] * 8,
        },
    }


# ---------- DeployConfig parsing ----------

class TestDeployConfigParsing:
    def test_round_trip_minimal(self):
        cfg = DeployConfig.model_validate(_minimal_deploy_yaml())
        assert cfg.ckpt.expected_unnorm_key == "libero_spatial_no_noops"
        assert cfg.proprio.adapt.output_dim == 8
        assert cfg.holdposition.gripper_native_midpoint == 0.5

    def test_proprio_source_total_dim_must_match_components(self):
        bad = _minimal_deploy_yaml()
        bad["proprio"]["source"]["total_dim"] = 99
        with pytest.raises(ValidationError, match="total_dim"):
            DeployConfig.model_validate(bad)

    def test_load_deploy_config_from_yaml_file(self, tmp_path):
        path = tmp_path / "v36.yaml"
        path.write_text(yaml.safe_dump(_minimal_deploy_yaml()))
        cfg = load_deploy_config(path)
        assert cfg.ckpt.expected_action_chunk_len == 8


# ---------- preprocess: JPEG decode + field mapping ----------

class TestPreprocess:
    def test_decode_jpeg_to_uint8_rgb_array(self):
        cfg = DeployConfig.model_validate(_minimal_deploy_yaml())
        adapter = DomainAdapter(cfg, _norm_stats_v36(), domain_id=0)
        req = PredictRequest(
            image_primary=_make_jpeg_b64(),
            image_wrist=_make_jpeg_b64(),
            proprio=[10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 0.5],
            instruction="x",
        )
        obs = adapter.preprocess(req)
        assert obs["scene_image"].dtype == np.uint8
        assert obs["scene_image"].shape == (224, 224, 3)
        assert obs["wrist_image"].shape == (224, 224, 3)
        assert obs["wrist_was_provided"] is True

    def test_wrist_absent_with_dropout_tolerant_zero_fills(self):
        """When deploy yaml allows wrist absent (Phase 0 wire_only_smoke=True
        bypasses hard-required check), zero-fill + wrist_was_provided=False."""
        cfg = DeployConfig.model_validate(_minimal_deploy_yaml())
        adapter = DomainAdapter(cfg, _norm_stats_v36(), domain_id=0)
        req = PredictRequest(
            image_primary=_make_jpeg_b64(),
            image_wrist=None,
            proprio=[0.0] * 7,
            instruction="x",
        )
        obs = adapter.preprocess(req)
        # In Phase 0 wire_only_smoke mode, missing wrist becomes a zero-image.
        assert obs["wrist_image"].shape == (224, 224, 3)
        np.testing.assert_array_equal(obs["wrist_image"], np.zeros((224, 224, 3), dtype=np.uint8))
        assert obs["wrist_was_provided"] is False


class TestProprioAdapt:
    def test_deg_to_rad_op(self):
        cfg = DeployConfig.model_validate(_minimal_deploy_yaml())
        # disable normalization for this unit-level test
        cfg = cfg.model_copy(update={"proprio": cfg.proprio.model_copy(
            update={"normalization": cfg.proprio.normalization.model_copy(update={"method": "none"})}
        )})
        adapter = DomainAdapter(cfg, norm_stats=None, domain_id=0)
        # 90 deg → π/2 rad ≈ 1.5708; gripper 0.5 → 0.5; pad_zeros → 0.0
        raw = [90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 0.5]
        out = adapter._apply_proprio_adapt(np.array(raw, dtype=np.float32))
        assert out.shape == (8,)
        np.testing.assert_allclose(out[:6], np.pi / 2, atol=1e-5)
        assert out[6] == 0.5
        assert out[7] == 0.0

    def test_q99_normalize_with_unit_stats_is_identity(self):
        cfg = DeployConfig.model_validate(_minimal_deploy_yaml())
        adapter = DomainAdapter(cfg, _norm_stats_v36(), domain_id=0)
        out = adapter._normalize_proprio(np.zeros(8, dtype=np.float32))
        # mean=0, q99=1, q01=-1 → normalize is identity at zero
        np.testing.assert_allclose(out, np.zeros(8), atol=1e-6)


# ---------- postprocess: gripper conv + frame conv + row-shape assert ----------

class TestPostprocess:
    def test_identity_gripper_conversion_when_native_eq_contract(self):
        """v36 native and SO-101 contract both normalized_0_1 closed=0/open=1
        → identity."""
        cfg = DeployConfig.model_validate(_minimal_deploy_yaml())
        adapter = DomainAdapter(cfg, _norm_stats_v36(), domain_id=0)
        native = np.array([[0.001, 0.0, 0.0, 0.0, 0.0, 0.0, 0.7]], dtype=np.float32)
        out = adapter.postprocess(native)
        assert out == [[pytest.approx(0.001), 0.0, 0.0, 0.0, 0.0, 0.0, pytest.approx(0.7)]]

    def test_signed_to_normalized_gripper_conversion(self):
        """signed_neg1_pos1 (open=-1, closed=+1) → normalized_0_1 (closed=0, open=1)."""
        cfg_d = _minimal_deploy_yaml()
        cfg_d["action"]["native"]["gripper"] = {
            "kind": "absolute",
            "units": "signed_neg1_pos1",
            "sign": {"closed": 1, "open": -1},
        }
        cfg = DeployConfig.model_validate(cfg_d)
        adapter = DomainAdapter(cfg, _norm_stats_v36(), domain_id=0)
        # native +1 (closed) → contract 0 (closed)
        # native -1 (open)   → contract 1 (open)
        # native  0 (mid)    → contract 0.5 (mid)
        native = np.array([
            [0, 0, 0, 0, 0, 0, +1.0],
            [0, 0, 0, 0, 0, 0, -1.0],
            [0, 0, 0, 0, 0, 0, 0.0],
        ], dtype=np.float32)
        out = adapter.postprocess(native)
        assert out[0][6] == pytest.approx(0.0)
        assert out[1][6] == pytest.approx(1.0)
        assert out[2][6] == pytest.approx(0.5)

    def test_row_width_assert_rejects_too_few_cols(self):
        cfg = DeployConfig.model_validate(_minimal_deploy_yaml())
        adapter = DomainAdapter(cfg, _norm_stats_v36(), domain_id=0)
        with pytest.raises(AssertionError, match="row width"):
            adapter.postprocess(np.zeros((1, 6), dtype=np.float32))

    def test_row_width_assert_rejects_too_many_cols(self):
        cfg = DeployConfig.model_validate(_minimal_deploy_yaml())
        adapter = DomainAdapter(cfg, _norm_stats_v36(), domain_id=0)
        with pytest.raises(AssertionError, match="row width"):
            adapter.postprocess(np.zeros((1, 8), dtype=np.float32))

    def test_q99_denormalize_respects_mask_false_passthrough(self):
        cfg_d = _minimal_deploy_yaml()
        cfg = DeployConfig.model_validate(cfg_d)
        # action.std=1 mean=0 except gripper (mask=False) — gripper dim untouched
        stats = _norm_stats_v36()
        stats["action"]["mean"] = [10.0] * 7  # would offset all dims if mask were True
        adapter = DomainAdapter(cfg, stats, domain_id=0)
        native = np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5]], dtype=np.float32)
        # gripper (mask=False) passes through unchanged; ee dims get +10 offset
        out = adapter.postprocess(native, denormalize=True)
        assert out[0][0] == pytest.approx(11.0)
        assert out[0][6] == pytest.approx(0.5)


# ---------- startup hard-fail assertions ----------

class TestStartupAssertions:
    def _norm_stats(self):
        return {"libero_spatial_no_noops": _norm_stats_v36()}

    def test_negative_domain_id_raises(self):
        cfg = DeployConfig.model_validate(_minimal_deploy_yaml())
        with pytest.raises(HardFailAssertion, match="domain_id"):
            DomainAdapter.validate_startup_xvla(
                cfg, meta_cfg={"model": {"num_domains": 1}, "data": {"unnorm_key": "libero_spatial_no_noops", "action_chunk_len": 8}},
                norm_stats=self._norm_stats(), domain_id=-1,
            )

    def test_domain_id_out_of_range_raises(self):
        cfg = DeployConfig.model_validate(_minimal_deploy_yaml())
        with pytest.raises(HardFailAssertion, match="domain_id"):
            DomainAdapter.validate_startup_xvla(
                cfg, meta_cfg={"model": {"num_domains": 1}, "data": {"unnorm_key": "libero_spatial_no_noops", "action_chunk_len": 8}},
                norm_stats=self._norm_stats(), domain_id=5,
            )

    def test_unnorm_key_mismatch_raises(self):
        cfg = DeployConfig.model_validate(_minimal_deploy_yaml())
        with pytest.raises(HardFailAssertion, match="unnorm_key"):
            DomainAdapter.validate_startup_xvla(
                cfg, meta_cfg={"model": {"num_domains": 1}, "data": {"unnorm_key": "libero_other", "action_chunk_len": 8}},
                norm_stats=self._norm_stats(), domain_id=0,
            )

    def test_action_chunk_len_fallback_chain_picks_default_8(self):
        """v36 sets cfg.data.action_chunk_len=8; v35-style ckpts set neither,
        falling back to C.ACTION_CHUNK_LEN=8."""
        cfg = DeployConfig.model_validate(_minimal_deploy_yaml())
        # cfg.data has no action_chunk_len → fallback to default
        DomainAdapter.validate_startup_xvla(
            cfg, meta_cfg={"model": {"num_domains": 1}, "data": {"unnorm_key": "libero_spatial_no_noops"}},
            norm_stats=self._norm_stats(), domain_id=0,
        )

    def test_holdposition_startup_skips_ckpt_asserts(self):
        cfg = DeployConfig.model_validate(_minimal_deploy_yaml())
        # no meta_cfg / no norm_stats → should still pass for hold_position mode
        DomainAdapter.validate_startup_hold_position(cfg, domain_id=0)
        with pytest.raises(HardFailAssertion, match="domain_id"):
            DomainAdapter.validate_startup_hold_position(cfg, domain_id=-1)

    def test_holdposition_startup_rejects_zero_chunk_len(self):
        bad = _minimal_deploy_yaml()
        bad["ckpt"]["expected_action_chunk_len"] = 0
        cfg = DeployConfig.model_validate(bad)
        with pytest.raises(HardFailAssertion, match="action_chunk_len"):
            DomainAdapter.validate_startup_hold_position(cfg, domain_id=0)

    def test_holdposition_startup_applies_gripper_compat(self):
        """Spec §Section 4 step 4 keeps gripper-convention compat check active
        in HoldPosition mode (only frame_conversion is skipped). Mismatched
        gripper conventions without wire_only_smoke must fail."""
        bad = _minimal_deploy_yaml()
        bad["wire_only_smoke"] = False
        bad["action"]["native"]["gripper"] = {
            "kind": "absolute",
            "units": "signed_neg1_pos1",
            "sign": {"closed": 1, "open": -1},
        }
        # contract still normalized_0_1; built-in linear remap will work, so
        # this should NOT fail (compat check passes for any (closed, open) pair).
        # The failure case is: sign.closed == sign.open (degenerate) — covered below.
        cfg = DeployConfig.model_validate(bad)
        DomainAdapter.validate_startup_hold_position(cfg, domain_id=0)  # ok

    def test_holdposition_startup_rejects_degenerate_native_gripper(self):
        bad = _minimal_deploy_yaml()
        bad["action"]["native"]["gripper"]["sign"] = {"closed": 0.5, "open": 0.5}
        cfg = DeployConfig.model_validate(bad)
        with pytest.raises(HardFailAssertion, match="gripper"):
            DomainAdapter.validate_startup_hold_position(cfg, domain_id=0)


class TestStartupAssertionsXVLAFull:
    """All hard-fail assertions from spec §Section 4 step 3, exercised against
    the xvla_adapter validator. Tests live in test_runtime_load.py per spec
    §Section 6 testing table — the implementations of the assertions are in
    DomainAdapter.validate_startup_xvla but the tests are gathered here for
    locality with the meta.json fixture."""

    def _ok_meta_cfg(self):
        return {"model": {"num_domains": 1}, "data": {"unnorm_key": "libero_spatial_no_noops", "action_chunk_len": 8}}

    def _ok_norm_stats(self):
        return {"libero_spatial_no_noops": _norm_stats_v36()}

    def test_action_dim_mismatch_raises(self):
        cfg_d = _minimal_deploy_yaml()
        cfg_d["ckpt"]["expected_action_dim"] = 9  # ckpt has 7
        cfg = DeployConfig.model_validate(cfg_d)
        with pytest.raises(HardFailAssertion, match="action_dim"):
            DomainAdapter.validate_startup_xvla(
                cfg, meta_cfg=self._ok_meta_cfg(),
                norm_stats=self._ok_norm_stats(), domain_id=0,
            )

    def test_proprio_dim_mismatch_raises(self):
        cfg_d = _minimal_deploy_yaml()
        cfg_d["ckpt"]["expected_proprio_dim"] = 9
        cfg_d["proprio"]["adapt"]["output_dim"] = 9
        cfg_d["proprio"]["adapt"]["steps"][-1]["dims"] = 2  # match output_dim
        cfg = DeployConfig.model_validate(cfg_d)
        with pytest.raises(HardFailAssertion, match="proprio"):
            DomainAdapter.validate_startup_xvla(
                cfg, meta_cfg=self._ok_meta_cfg(),
                norm_stats=self._ok_norm_stats(), domain_id=0,
            )

    def test_proprio_adapt_output_dim_mismatch_raises(self):
        cfg_d = _minimal_deploy_yaml()
        cfg_d["proprio"]["adapt"]["output_dim"] = 9
        cfg_d["proprio"]["adapt"]["steps"][-1]["dims"] = 2
        cfg = DeployConfig.model_validate(cfg_d)
        with pytest.raises(HardFailAssertion, match="output_dim"):
            DomainAdapter.validate_startup_xvla(
                cfg, meta_cfg=self._ok_meta_cfg(),
                norm_stats=self._ok_norm_stats(), domain_id=0,
            )

    def test_hard_required_wrist_missing_raises(self):
        cfg_d = _minimal_deploy_yaml()
        cfg_d["request_fields"]["wrist_image"] = None
        cfg = DeployConfig.model_validate(cfg_d)
        meta_cfg = self._ok_meta_cfg()
        meta_cfg["model"]["use_wrist_bridge"] = True  # hard-required
        with pytest.raises(HardFailAssertion, match="wrist"):
            DomainAdapter.validate_startup_xvla(
                cfg, meta_cfg=meta_cfg,
                norm_stats=self._ok_norm_stats(), domain_id=0,
            )

    def test_frame_mismatch_without_wire_only_smoke_raises(self):
        cfg_d = _minimal_deploy_yaml()
        cfg_d["wire_only_smoke"] = False  # no escape hatch
        # native.frame=world, contract.frame=ee_local in default → mismatched
        cfg = DeployConfig.model_validate(cfg_d)
        with pytest.raises(HardFailAssertion, match="frame"):
            DomainAdapter.validate_startup_xvla(
                cfg, meta_cfg=self._ok_meta_cfg(),
                norm_stats=self._ok_norm_stats(), domain_id=0,
            )
```

- [ ] **Step 2: Run → verify ImportError**

```bash
uv run pytest tests/test_domain_adapter.py -v
```

- [ ] **Step 3: Implement DomainAdapter + DeployConfig**

```python
# src/vla_project/deployment/domain_adapter.py
"""DomainAdapter — per-domain in/out conversion + DeployConfig pydantic.

Loads `configs/deploy/<robot>_<model>.yaml` into a typed DeployConfig and
provides:
  - preprocess(req): JPEG decode + field-name mapping + proprio adapt
    (deg_to_rad, copy, pad_zeros) + Q99 normalize → Obs dict
  - postprocess(native_chunk): gripper-convention conversion + frame
    conversion (none in Phase 0) + row-shape assert → list[list[float]]
  - validate_startup_xvla / validate_startup_hold_position: hard-fail asserts
    per spec §Section 4 startup validation flow

Phase 0 implements the contract for HoldPosition path; xvla_adapter mode
runs all the same code paths but the predictor itself is stubbed.
"""
from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import Any, Literal

import numpy as np
import yaml
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field, model_validator

# C.ACTION_CHUNK_LEN — single source of truth for the v33/v35 default
from vla_project.data import constants as C

from vla_project.deployment.schemas import PredictRequest


class HardFailAssertion(Exception):
    """Raised at startup if deploy yaml ↔ ckpt metadata ↔ args are inconsistent."""


# ---------- DeployConfig pydantic schema ----------

class _ProprioComponent(BaseModel):
    name: str
    dims: int
    units: str


class _ProprioSource(BaseModel):
    components: list[_ProprioComponent]
    total_dim: int

    @model_validator(mode="after")
    def _check_total(self) -> "_ProprioSource":
        if sum(c.dims for c in self.components) != self.total_dim:
            raise ValueError("proprio.source.total_dim must equal sum(components.dims)")
        return self


class _ProprioStep(BaseModel):
    model_config = ConfigDict(extra="forbid")
    op: Literal["deg_to_rad", "rad_to_deg", "copy", "pad_zeros", "scale", "constant"]
    source: str | None = None
    dims: int = 1
    value: float | None = None  # for "constant"
    factor: float | None = None  # for "scale"


class _ProprioAdapt(BaseModel):
    steps: list[_ProprioStep]
    output_dim: int


class _ProprioNormalization(BaseModel):
    method: Literal["none", "q99"] = "q99"
    stats_key: str = "proprio"


class _Proprio(BaseModel):
    source: _ProprioSource
    adapt: _ProprioAdapt
    normalization: _ProprioNormalization


class _GripperSign(BaseModel):
    closed: float
    open: float


class _Gripper(BaseModel):
    kind: Literal["absolute", "delta", "binary"]
    units: Literal["normalized_0_1", "signed_neg1_pos1", "binary_threshold_0p5"]
    sign: _GripperSign


class _ActionSide(BaseModel):
    units: Literal["meter_axisangle_rad"] = "meter_axisangle_rad"
    frame: Literal["ee_local", "world"]
    gripper: _Gripper


class _Denormalization(BaseModel):
    method: Literal["none", "q99", "mean_std"] = "q99"
    stats_key: str = "action"


class _FrameConversion(BaseModel):
    method: Literal["none", "world_to_ee_local", "ee_local_to_world"] = "none"


class _Action(BaseModel):
    native: _ActionSide
    contract: _ActionSide
    denormalization: _Denormalization
    frame_conversion: _FrameConversion


class _CkptIdentity(BaseModel):
    expected_unnorm_key: str
    expected_action_chunk_len: int
    expected_action_dim: int
    expected_proprio_dim: int


class _RequestFields(BaseModel):
    scene_image: str
    wrist_image: str | None = None
    proprio: str = "proprio"
    instruction: str = "instruction"


class _HoldPosition(BaseModel):
    gripper_native_midpoint: float = 0.5


class _Runtime(BaseModel):
    device: str = "cuda:0"
    dtype: Literal["bf16", "fp32"] = "bf16"
    torch_compile: Literal["off", "reduce-overhead", "default"] = "off"
    warmup_iters: int = 1


class DeployConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    ckpt: _CkptIdentity
    request_fields: _RequestFields
    proprio: _Proprio
    action: _Action
    holdposition: _HoldPosition = Field(default_factory=_HoldPosition)
    wire_only_smoke: bool = False
    runtime: _Runtime = Field(default_factory=_Runtime)


def load_deploy_config(path: str | Path) -> DeployConfig:
    return DeployConfig.model_validate(yaml.safe_load(Path(path).read_text()))


# ---------- DomainAdapter ----------

class DomainAdapter:
    def __init__(
        self,
        cfg: DeployConfig,
        norm_stats: dict | None,
        domain_id: int,
        *,
        wrist_hard_required: bool = False,
    ) -> None:
        self.cfg = cfg
        self.norm_stats = norm_stats
        self.domain_id = int(domain_id)
        self.wrist_hard_required = bool(wrist_hard_required)

    # ----- preprocess -----

    def preprocess(self, req: PredictRequest) -> dict[str, Any]:
        scene = self._decode_jpeg_b64(req.image_primary)
        wrist_b64 = req.image_wrist
        if wrist_b64 is not None:
            wrist = self._decode_jpeg_b64(wrist_b64)
            wrist_was_provided = True
        else:
            if self.wrist_hard_required:
                raise ValueError(
                    "checkpoint requires wrist_image (use_wrist_bridge or "
                    "DINOv2 path) but request omitted it"
                )
            wrist = np.zeros((224, 224, 3), dtype=np.uint8)
            wrist_was_provided = False
        proprio_raw = np.asarray(req.proprio, dtype=np.float32)
        if proprio_raw.shape[0] != self.cfg.proprio.source.total_dim:
            raise ValueError(
                f"proprio length {proprio_raw.shape[0]} != "
                f"deploy.proprio.source.total_dim {self.cfg.proprio.source.total_dim}"
            )
        proprio_adapted = self._apply_proprio_adapt(proprio_raw)
        proprio_normalized = self._normalize_proprio(proprio_adapted)
        return {
            "scene_image": scene,
            "wrist_image": wrist,
            "wrist_was_provided": wrist_was_provided,
            "proprio": proprio_normalized,
            "language": req.instruction,
        }

    @staticmethod
    def _decode_jpeg_b64(b64_str: str) -> np.ndarray:
        raw = base64.b64decode(b64_str)
        img = Image.open(io.BytesIO(raw)).convert("RGB")
        return np.asarray(img, dtype=np.uint8)

    def _apply_proprio_adapt(self, raw: np.ndarray) -> np.ndarray:
        # Index source components by name for "source: <name>" lookups.
        offsets: dict[str, tuple[int, int]] = {}
        i = 0
        for c in self.cfg.proprio.source.components:
            offsets[c.name] = (i, i + c.dims)
            i += c.dims
        out_parts: list[np.ndarray] = []
        for step in self.cfg.proprio.adapt.steps:
            if step.op == "deg_to_rad":
                lo, hi = offsets[step.source]  # type: ignore[index]
                out_parts.append(raw[lo:hi] * np.float32(np.pi / 180.0))
            elif step.op == "rad_to_deg":
                lo, hi = offsets[step.source]  # type: ignore[index]
                out_parts.append(raw[lo:hi] * np.float32(180.0 / np.pi))
            elif step.op == "copy":
                lo, hi = offsets[step.source]  # type: ignore[index]
                out_parts.append(raw[lo:hi].copy())
            elif step.op == "pad_zeros":
                out_parts.append(np.zeros(step.dims, dtype=np.float32))
            elif step.op == "scale":
                lo, hi = offsets[step.source]  # type: ignore[index]
                out_parts.append(raw[lo:hi] * np.float32(step.factor or 1.0))
            elif step.op == "constant":
                out_parts.append(np.full(step.dims, step.value or 0.0, dtype=np.float32))
            else:
                raise ValueError(f"unknown proprio.adapt op: {step.op}")
        out = np.concatenate(out_parts, axis=0).astype(np.float32)
        if out.shape[0] != self.cfg.proprio.adapt.output_dim:
            raise ValueError(
                f"proprio.adapt produced {out.shape[0]} dims, expected "
                f"output_dim={self.cfg.proprio.adapt.output_dim}"
            )
        return out

    def _normalize_proprio(self, x: np.ndarray) -> np.ndarray:
        if self.cfg.proprio.normalization.method == "none" or self.norm_stats is None:
            return x
        # Q99: normalize each dim into [-1, +1] using (q01, q99) with mask.
        stats = self.norm_stats[self.cfg.proprio.normalization.stats_key]
        q01 = np.asarray(stats["q01"], dtype=np.float32)
        q99 = np.asarray(stats["q99"], dtype=np.float32)
        mask = np.asarray(stats.get("mask", [True] * len(q01)), dtype=bool)
        span = q99 - q01
        span = np.where(span == 0, 1.0, span)
        normed = 2.0 * (x - q01) / span - 1.0
        # Clamp to [-1, +1] (training-time q99 clipping convention).
        normed = np.clip(normed, -1.0, 1.0)
        return np.where(mask, normed, x).astype(np.float32)

    # ----- postprocess -----

    def postprocess(self, native_chunk: np.ndarray, *, denormalize: bool = False) -> list[list[float]]:
        a = np.asarray(native_chunk, dtype=np.float32)
        assert a.ndim == 2, f"native_chunk must be 2-D; got {a.ndim}-D"
        T, A = a.shape
        # row-shape assert (per spec §Section 6 test_domain_adapter expectations)
        if A != self.cfg.ckpt.expected_action_dim:
            raise AssertionError(f"row width {A} != expected_action_dim {self.cfg.ckpt.expected_action_dim}")

        # Optional denorm (used when called from XVLAAdapter path; HoldPosition skips).
        if denormalize and self.cfg.action.denormalization.method == "q99" and self.norm_stats is not None:
            a = self._q99_denorm_action(a)

        # Frame conversion: Phase 0 only supports "none". Implementation deferred to Phase 1.
        if self.cfg.action.frame_conversion.method != "none":
            raise NotImplementedError(
                f"frame_conversion.method={self.cfg.action.frame_conversion.method} "
                "is Phase 1 work"
            )

        # Gripper conversion (linear remap based on (closed, open) in native + contract).
        a = self._convert_gripper(a)
        return a.tolist()

    def _q99_denorm_action(self, a: np.ndarray) -> np.ndarray:
        stats = self.norm_stats[self.cfg.action.denormalization.stats_key]  # type: ignore[index]
        q01 = np.asarray(stats["q01"], dtype=np.float32)
        q99 = np.asarray(stats["q99"], dtype=np.float32)
        mean = np.asarray(stats["mean"], dtype=np.float32)
        mask = np.asarray(stats.get("mask", [True] * a.shape[1]), dtype=bool)
        span = q99 - q01
        span = np.where(span == 0, 1.0, span)
        # Q99 uses mean+std, but the spec aligns with the existing
        # denormalize_action_q99 path which uses (q01, q99) + mean. We mirror
        # that: physical = mean + arr * (q99 - q01) / 2 for masked dims; pass
        # through unchanged for mask=False dims.
        denormed = mean + a * (span / 2.0)
        return np.where(mask, denormed, a).astype(np.float32)

    def _convert_gripper(self, a: np.ndarray) -> np.ndarray:
        n = self.cfg.action.native.gripper
        c = self.cfg.action.contract.gripper
        # Identity short-circuit
        if (n.units == c.units and n.sign.closed == c.sign.closed and n.sign.open == c.sign.open):
            return a
        # Linear remap: t in [0, 1] is the "openness" fraction.
        denom = (n.sign.open - n.sign.closed)
        if denom == 0:
            raise ValueError("native.gripper sign.open == sign.closed — cannot remap")
        g_native = a[:, -1]
        t = (g_native - n.sign.closed) / denom
        g_contract = c.sign.closed + t * (c.sign.open - c.sign.closed)
        out = a.copy()
        out[:, -1] = g_contract
        return out

    # ----- startup validators -----

    @staticmethod
    def validate_startup_xvla(
        cfg: DeployConfig,
        *,
        meta_cfg: dict,
        norm_stats: dict,
        domain_id: int,
    ) -> None:
        m_model = meta_cfg.get("model", {})
        m_data = meta_cfg.get("data", {})

        num_domains = int(m_model.get("num_domains", 0))
        if not (0 <= domain_id < num_domains):
            raise HardFailAssertion(
                f"domain_id={domain_id} out of range [0, {num_domains})"
            )
        if m_data.get("unnorm_key") != cfg.ckpt.expected_unnorm_key:
            raise HardFailAssertion(
                f"ckpt unnorm_key={m_data.get('unnorm_key')!r} != "
                f"expected {cfg.ckpt.expected_unnorm_key!r}"
            )
        # action_chunk_len fallback chain (per spec §Section 4 step 3)
        resolved_chunk_len = (
            m_model.get("action_chunk_len")
            or m_data.get("action_chunk_len")
            or C.ACTION_CHUNK_LEN
        )
        if resolved_chunk_len != cfg.ckpt.expected_action_chunk_len:
            raise HardFailAssertion(
                f"resolved action_chunk_len={resolved_chunk_len} != "
                f"expected {cfg.ckpt.expected_action_chunk_len}"
            )
        unk = cfg.ckpt.expected_unnorm_key
        action_mean = norm_stats[unk]["action"]["mean"]
        if len(action_mean) != cfg.ckpt.expected_action_dim:
            raise HardFailAssertion(
                f"len(norm_stats.action.mean)={len(action_mean)} != "
                f"expected_action_dim={cfg.ckpt.expected_action_dim}"
            )
        proprio_mean = norm_stats[unk]["proprio"]["mean"]
        if len(proprio_mean) != cfg.ckpt.expected_proprio_dim:
            raise HardFailAssertion(
                f"len(norm_stats.proprio.mean)={len(proprio_mean)} != "
                f"expected_proprio_dim={cfg.ckpt.expected_proprio_dim}"
            )
        if cfg.proprio.adapt.output_dim != cfg.ckpt.expected_proprio_dim:
            raise HardFailAssertion(
                f"deploy.proprio.adapt.output_dim={cfg.proprio.adapt.output_dim} != "
                f"expected_proprio_dim={cfg.ckpt.expected_proprio_dim}"
            )
        # Frame compatibility (hard-fail unless wire_only_smoke=True).
        if (
            cfg.action.native.frame != cfg.action.contract.frame
            and cfg.action.frame_conversion.method == "none"
            and not cfg.wire_only_smoke
        ):
            raise HardFailAssertion(
                f"native.frame={cfg.action.native.frame!r} != contract.frame="
                f"{cfg.action.contract.frame!r} with frame_conversion=none. "
                "Set wire_only_smoke=true to bypass for smoke testing."
            )
        # Wrist requirement (hard / soft path, per spec §Section 5 line 354)
        bridge_or_dinov2 = (
            m_model.get("use_wrist_bridge", False)
            or m_model.get("use_scene_wrist_dinov2_llm", False)
            or m_model.get("wrist_dinov2", False)
        )
        wrist_in_llm = m_model.get("wrist_in_llm", False)
        dropout = float(m_model.get("wrist_view_dropout_p") or 0.0)
        wrist_field = cfg.request_fields.wrist_image
        if bridge_or_dinov2 and not wrist_field:
            raise HardFailAssertion(
                "ckpt requires wrist (use_wrist_bridge or DINOv2 path); "
                "deploy.request_fields.wrist_image must be set"
            )
        if wrist_in_llm and dropout == 0.0 and not wrist_field:
            raise HardFailAssertion(
                "ckpt requires wrist (wrist_in_llm with no dropout); "
                "deploy.request_fields.wrist_image must be set"
            )

    @staticmethod
    def validate_startup_hold_position(
        cfg: DeployConfig,
        *,
        domain_id: int,
    ) -> None:
        if domain_id < 0:
            raise HardFailAssertion(
                f"domain_id={domain_id} must be >= 0 (upper bound only "
                "checkable in xvla_adapter mode)"
            )
        # deploy-yaml-internal asserts only; ckpt-derived asserts skipped.
        if cfg.proprio.source.total_dim != sum(c.dims for c in cfg.proprio.source.components):
            raise HardFailAssertion("proprio.source.total_dim != sum(components.dims)")
        if cfg.ckpt.expected_action_dim != 7:
            raise HardFailAssertion(
                f"expected_action_dim={cfg.ckpt.expected_action_dim} != 7 (MVP fixed)"
            )
        if cfg.ckpt.expected_action_chunk_len <= 0:
            raise HardFailAssertion(
                f"expected_action_chunk_len={cfg.ckpt.expected_action_chunk_len} must be > 0"
            )
        # Gripper compat (linear remap requires non-degenerate native sign).
        n = cfg.action.native.gripper
        if n.sign.open == n.sign.closed:
            raise HardFailAssertion(
                f"native.gripper.sign.open == sign.closed ({n.sign.open}); "
                "gripper remap is degenerate"
            )
```

- [ ] **Step 4: Run tests → all pass**

```bash
uv run pytest tests/test_domain_adapter.py -v
```

Expected: all tests PASS.

- [ ] **Step 5: codex review (focused; this task is the largest)**

```bash
codex exec -m gpt-5.5 --skip-git-repo-check 'Review src/vla_project/deployment/domain_adapter.py + tests/test_domain_adapter.py. Implements spec §Section 4 deploy YAML schema + §Section 4 startup validation + §Section 3 preprocess/postprocess. Concerns to flag: (a) Q99 denorm formula — does (mean + a * span/2) match training-time normalize_action_q99 (check vla_project/data/normalization.py)? (b) gripper linear remap formula correctness for sign-flipped conventions. (c) wrist requirement split (bridge vs dropout-tolerant) at validate_startup_xvla — does it match spec §Section 5 line 354? (d) any startup assertion missing from spec §Section 4 step 3? Reply terse, line numbers.'
```

- [ ] **Step 6: Commit**

```bash
git add src/vla_project/deployment/domain_adapter.py tests/test_domain_adapter.py
git commit -m "$(cat <<'EOF'
feat(deployment): DomainAdapter + DeployConfig schema

Per-domain in/out conversion driven by configs/deploy/<robot>_<model>.yaml.
- DeployConfig pydantic v2 schema with full sub-model decomposition
- preprocess: JPEG decode, field-name mapping, proprio.adapt step ops
  (deg_to_rad, copy, pad_zeros, scale, constant), Q99 normalization
- postprocess: gripper convention conversion (identity + linear remap),
  frame_conversion=none enforced (Phase 1 implements other modes), row-shape
  assert per spec §Section 6 testing requirements
- validate_startup_xvla / validate_startup_hold_position: hard-fail asserts
  matching spec §Section 4 step 3, including wrist-requirement split

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: ModelRuntime stub + tests

**Files:**
- Create: `src/vla_project/deployment/runtime.py`
- Create: `tests/test_runtime_load.py`

Phase 0 stub: `from_export` loads `meta.json` and validates structure; `__call__` raises `NotImplementedError`.

- [ ] **Step 1: Failing tests**

```python
# tests/test_runtime_load.py
"""ModelRuntime stub: loads meta.json + provides startup validation hooks.
Full forward path is Phase 1.

Also covers the build_app() startup assertion errors that are wired through
domain_adapter.validate_startup_xvla; this file focuses on the runtime side
(meta loading + paths)."""
import json

import pytest

from vla_project.deployment.runtime import ModelRuntime, MetaJsonError


def _write_meta(tmp_path, payload):
    p = tmp_path / "meta.json"
    p.write_text(json.dumps(payload))
    return p


def test_from_export_missing_meta_json_raises(tmp_path):
    with pytest.raises(MetaJsonError, match="meta.json"):
        ModelRuntime.from_export(tmp_path)


def test_from_export_loads_step_and_cfg_norm_stats(tmp_path):
    _write_meta(tmp_path, {
        "step": 15000,
        "cfg": {"model": {"num_domains": 1}, "data": {"unnorm_key": "k"}},
        "norm_stats": {"k": {"action": {}, "proprio": {}}},
    })
    rt = ModelRuntime.from_export(tmp_path)
    assert rt.step == 15000
    assert rt.cfg["model"]["num_domains"] == 1
    assert "k" in rt.norm_stats


def test_call_raises_not_implemented_in_phase_0(tmp_path):
    _write_meta(tmp_path, {
        "step": 0,
        "cfg": {"model": {"num_domains": 1}, "data": {"unnorm_key": "k"}},
        "norm_stats": {"k": {}},
    })
    rt = ModelRuntime.from_export(tmp_path)
    with pytest.raises(NotImplementedError, match="Phase 1"):
        rt({})
```

- [ ] **Step 2: Run → ImportError**

```bash
uv run pytest tests/test_runtime_load.py -v
```

- [ ] **Step 3: Implement stub**

```python
# src/vla_project/deployment/runtime.py
"""ModelRuntime — Phase 0 stub.

Phase 0: loads meta.json (so xvla_adapter startup validation can run) but
__call__(batch) raises NotImplementedError. Phase 1 fills in the torch
forward path per spec §Section 5 ModelRuntime.

The classmethod from_export(ckpt_dir) is the canonical entry; tests assert
its behavior on synthetic meta.json fixtures.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class MetaJsonError(Exception):
    """meta.json missing, malformed, or missing required keys."""


class ModelRuntime:
    def __init__(self, *, step: int, cfg: dict, norm_stats: dict, ckpt_dir: Path) -> None:
        self.step = step
        self.cfg = cfg
        self.norm_stats = norm_stats
        self.ckpt_dir = ckpt_dir

    @classmethod
    def from_export(
        cls,
        ckpt_dir: str | Path,
        *,
        device: str = "cuda:0",
        dtype: str = "bf16",
        torch_compile: str = "off",
        warmup_iters: int = 1,
    ) -> "ModelRuntime":
        ckpt_dir = Path(ckpt_dir)
        meta_path = ckpt_dir / "meta.json"
        if not meta_path.is_file():
            raise MetaJsonError(f"missing meta.json under {ckpt_dir}")
        meta = json.loads(meta_path.read_text())
        for required_key in ("step", "cfg", "norm_stats"):
            if required_key not in meta:
                raise MetaJsonError(f"meta.json missing required key {required_key!r}")
        # Phase 0 ignores device / dtype / torch_compile / warmup_iters; Phase 1 wires them.
        _ = (device, dtype, torch_compile, warmup_iters)
        return cls(
            step=int(meta["step"]),
            cfg=meta["cfg"],
            norm_stats=meta["norm_stats"],
            ckpt_dir=ckpt_dir,
        )

    def __call__(self, batch: dict[str, Any]) -> Any:
        raise NotImplementedError(
            "ModelRuntime forward path is Phase 1 work. "
            "See docs/superpowers/specs/2026-05-06-vla-inference-server-design.md "
            "§Section 5 ModelRuntime."
        )
```

- [ ] **Step 4: Pass + codex review + commit (use the same pattern as Tasks 2-5)**

```bash
uv run pytest tests/test_runtime_load.py -v
codex exec -m gpt-5.5 --skip-git-repo-check 'Review src/vla_project/deployment/runtime.py + tests/test_runtime_load.py. Phase 0 stub: meta.json load + NotImplementedError on __call__. Spec §Section 5 ModelRuntime full impl is Phase 1. Concerns: any missed required key in meta.json (cross-check spec §Section 4 export format)? Reply terse.'
git add src/vla_project/deployment/runtime.py tests/test_runtime_load.py
git commit -m "$(cat <<'EOF'
feat(deployment): ModelRuntime stub for Phase 0

Loads meta.json so DomainAdapter.validate_startup_xvla can run, but
__call__ raises NotImplementedError("Phase 1") — full forward path is
deferred until v36 ckpt is trained.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: XVLAAdapterChunkPredictor stub + tests

**Files:**
- Create: `src/vla_project/deployment/predictors/xvla_adapter.py`
- Create: `tests/test_predictor_xvla_adapter.py`

- [ ] **Step 1: Failing tests**

```python
# tests/test_predictor_xvla_adapter.py
"""Phase 0: XVLAAdapterChunkPredictor is a typed shell that raises
NotImplementedError on predict(). Phase 1 fills in the SigLIP transform +
tokenize + batch build + Q99 denorm pipeline per spec §Section 5.

This file ensures the constructor signature matches what the spec describes
so the Phase 1 implementer cannot drift the API."""
import pytest

from vla_project.deployment.predictors.xvla_adapter import XVLAAdapterChunkPredictor


def test_construction_takes_documented_args():
    """All ctor args from spec §Section 5 line 466 — signature freeze."""
    p = XVLAAdapterChunkPredictor(
        runtime=None,         # Phase 1 will be ModelRuntime
        tokenizer=None,
        image_transform=None,
        action_q99=None,
        action_chunk_len=8,
        action_dim=7,
        domain_id=0,
    )
    assert p.chunk_len == 8
    assert p.action_dim == 7


def test_predict_raises_not_implemented_in_phase_0():
    p = XVLAAdapterChunkPredictor(
        runtime=None, tokenizer=None, image_transform=None,
        action_q99=None, action_chunk_len=8, action_dim=7, domain_id=0,
    )
    with pytest.raises(NotImplementedError, match="Phase 1"):
        p.predict({})
```

- [ ] **Step 2: Run → ImportError**

- [ ] **Step 3: Implement stub**

```python
# src/vla_project/deployment/predictors/xvla_adapter.py
"""XVLAAdapterChunkPredictor — Phase 0 typed shell.

Constructor signature is frozen per spec §Section 5 line 466 so Phase 1
cannot drift the public API. predict() raises NotImplementedError.

Phase 1 implementation will follow XVLAAdapterPolicy._refill_buffer for
the forward path (SigLIP transform + tokenize + batch build, including
DINOv2 conditional keys + wrist_was_provided plumbing per spec §Section 5).
"""
from __future__ import annotations

from typing import Any

import numpy as np

from vla_project.deployment.predictors.base import ChunkPredictor


class XVLAAdapterChunkPredictor(ChunkPredictor):
    def __init__(
        self,
        runtime: Any,                  # Phase 1: ModelRuntime
        tokenizer: Any,                # Phase 1: GemmaPromptTokenizer
        image_transform: Any,          # Phase 1: SiglipImageTransform
        action_q99: Any,               # Phase 1: Q99Stats from meta.norm_stats
        action_chunk_len: int,
        action_dim: int,
        domain_id: int,
    ) -> None:
        self._T = int(action_chunk_len)
        self._A = int(action_dim)
        self.runtime = runtime
        self.tokenizer = tokenizer
        self.image_transform = image_transform
        self.action_q99 = action_q99
        self.domain_id = int(domain_id)

    @property
    def chunk_len(self) -> int:
        return self._T

    @property
    def action_dim(self) -> int:
        return self._A

    def predict(self, obs: dict[str, Any]) -> np.ndarray:
        raise NotImplementedError(
            "XVLAAdapterChunkPredictor.predict() is Phase 1 work. "
            "See spec §Section 5 lines 478-504."
        )
```

- [ ] **Step 4-6: Pass tests + codex + commit (same pattern)**

```bash
uv run pytest tests/test_predictor_xvla_adapter.py -v
codex exec -m gpt-5.5 --skip-git-repo-check 'Review predictors/xvla_adapter.py + test. Phase 0 = signature freeze + NotImplementedError. Spec §Section 5 line 466 ctor args verbatim. Concerns: any ctor arg missing or extra? predict() docstring point at the right spec lines for Phase 1 implementer? Reply terse.'
git add src/vla_project/deployment/predictors/xvla_adapter.py tests/test_predictor_xvla_adapter.py
git commit -m "feat(deployment): XVLAAdapterChunkPredictor stub (signature frozen for Phase 1)

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 8: inference_server (build_app + /predict + /healthz + slow-path log)

**Files:**
- Create: `src/vla_project/deployment/inference_server.py`
- Modify: `src/vla_project/deployment/__init__.py` (export `build_app`)

**Task ordering:** Execute **Task 10 (deploy yamls) before this task** so the minimal TDD test below has a real `configs/deploy/v36_libero_spatial.yaml` to load. Re-order in your execution if proceeding linearly.

- [ ] **Step 1: Write the failing minimal TDD test FIRST**

```python
# tests/test_inference_server_minimal.py
"""Minimal TDD test for build_app — full smoke is in test_serve_smoke.py
(Task 11). This file enforces TDD discipline for Task 8 itself: the
build_app entry must fail import before the implementation lands."""
from fastapi.testclient import TestClient

from vla_project.deployment.inference_server import build_app


def test_build_app_returns_fastapi_with_healthz():
    app = build_app(
        predictor_kind="hold_position",
        checkpoint=None,
        deploy_config_path="configs/deploy/v36_libero_spatial.yaml",
        domain_id=0,
        inject_sleep_s=0.0,
    )
    client = TestClient(app)
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
```

Run:
```bash
uv run pytest tests/test_inference_server_minimal.py -v
```

Expected: ImportError on `build_app` (module does not yet exist).

- [ ] **Step 2: Implement build_app — verify the minimal test passes after the implementation lands**

```python
# src/vla_project/deployment/inference_server.py
"""FastAPI app factory + /predict + /healthz routes.

Reads deploy yaml + (optionally) ckpt meta.json, constructs DomainAdapter
and ChunkPredictor, mounts the FastAPI app. See spec §Section 6 for
HTTP code mapping, observability fields, and Phase 0 acceptance gate.
"""
from __future__ import annotations

import json
import logging
import time
import uuid
from pathlib import Path
from typing import Literal

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from vla_project.deployment.domain_adapter import (
    DomainAdapter,
    HardFailAssertion,
    load_deploy_config,
)
from vla_project.deployment.predictors.base import ChunkPredictor
from vla_project.deployment.predictors.hold_position import HoldPositionChunkPredictor
from vla_project.deployment.predictors.xvla_adapter import XVLAAdapterChunkPredictor
from vla_project.deployment.runtime import ModelRuntime
from vla_project.deployment.schemas import PredictRequest, PredictResponse


logger = logging.getLogger("vla_project.deployment")


_LATENCY_BUDGET_MS = 266.0  # spec §Section 3 latency budget breakdown


def build_app(
    *,
    predictor_kind: Literal["hold_position", "xvla_adapter"],
    checkpoint: str | Path | None,
    deploy_config_path: str | Path,
    domain_id: int,
    inject_sleep_s: float = 0.0,
) -> FastAPI:
    cfg = load_deploy_config(deploy_config_path)

    runtime: ModelRuntime | None = None
    norm_stats: dict | None = None

    if predictor_kind == "xvla_adapter":
        if checkpoint is None:
            raise ValueError("--checkpoint required when predictor_kind=xvla_adapter")
        runtime = ModelRuntime.from_export(
            checkpoint,
            device=cfg.runtime.device,
            dtype=cfg.runtime.dtype,
            torch_compile=cfg.runtime.torch_compile,
            warmup_iters=cfg.runtime.warmup_iters,
        )
        norm_stats = runtime.norm_stats
        DomainAdapter.validate_startup_xvla(
            cfg,
            meta_cfg=runtime.cfg,
            norm_stats=norm_stats,
            domain_id=domain_id,
        )
    else:
        DomainAdapter.validate_startup_hold_position(cfg, domain_id=domain_id)

    # Compute wrist_hard_required from the loaded ckpt cfg (None for hold_position).
    wrist_hard_required = False
    if predictor_kind == "xvla_adapter" and runtime is not None:
        m_model = runtime.cfg.get("model", {})
        wrist_hard_required = bool(
            m_model.get("use_wrist_bridge", False)
            or m_model.get("use_scene_wrist_dinov2_llm", False)
            or m_model.get("wrist_dinov2", False)
            or (m_model.get("wrist_in_llm", False) and float(m_model.get("wrist_view_dropout_p") or 0.0) == 0.0)
        )

    adapter = DomainAdapter(
        cfg=cfg,
        norm_stats=(norm_stats[cfg.ckpt.expected_unnorm_key] if norm_stats else None),
        domain_id=domain_id,
        wrist_hard_required=wrist_hard_required,
    )

    predictor: ChunkPredictor
    if predictor_kind == "hold_position":
        predictor = HoldPositionChunkPredictor(
            chunk_len=cfg.ckpt.expected_action_chunk_len,
            action_dim=cfg.ckpt.expected_action_dim,
            gripper_native_midpoint=cfg.holdposition.gripper_native_midpoint,
        )
    else:
        predictor = XVLAAdapterChunkPredictor(
            runtime=runtime,
            tokenizer=None,                 # Phase 1
            image_transform=None,           # Phase 1
            action_q99=norm_stats[cfg.ckpt.expected_unnorm_key]["action"] if norm_stats else None,
            action_chunk_len=cfg.ckpt.expected_action_chunk_len,
            action_dim=cfg.ckpt.expected_action_dim,
            domain_id=domain_id,
        )

    # ---- FastAPI app ----
    app = FastAPI(title="X-VLA-Adapter Inference Server")
    state_ready_at_ns = time.monotonic_ns()
    state = {
        "predictor_kind": predictor_kind,
        "predictor_class": type(predictor).__name__,
        "ready_at_ns": state_ready_at_ns,
        "inject_sleep_s": float(inject_sleep_s),
    }

    @app.get("/healthz")
    async def healthz() -> dict:
        return {
            "status": "ok",
            "predictor": state["predictor_class"],
            "ready_at_ns": state["ready_at_ns"],
        }

    @app.post("/predict", response_model=PredictResponse)
    async def predict(req: PredictRequest, request: Request) -> PredictResponse:
        request_id = request.headers.get("X-Request-Id") or uuid.uuid4().hex
        t0 = time.monotonic_ns()
        outcome: str = "ok"
        error_class: str | None = None
        error_msg: str | None = None
        try:
            obs = adapter.preprocess(req)
        except (ValueError, AssertionError) as e:
            outcome = "invalid_request"
            error_class = type(e).__name__
            error_msg = str(e)
            elapsed_ms = (time.monotonic_ns() - t0) / 1e6
            _log_request(request_id, elapsed_ms, state, outcome, error_class, error_msg)
            raise HTTPException(status_code=422, detail=str(e)) from e

        # Optional injected sleep for slow-path smoke (test-only).
        if state["inject_sleep_s"] > 0:
            import asyncio
            await asyncio.sleep(state["inject_sleep_s"])

        try:
            native = predictor.predict(obs)
        except NotImplementedError as e:
            outcome = "predictor_error"
            error_class = "NotImplementedError"
            error_msg = str(e)
            elapsed_ms = (time.monotonic_ns() - t0) / 1e6
            _log_request(request_id, elapsed_ms, state, outcome, error_class, error_msg)
            raise HTTPException(status_code=500, detail=str(e)) from e
        except Exception as e:  # noqa: BLE001
            outcome = "predictor_error"
            error_class = type(e).__name__
            error_msg = str(e)
            elapsed_ms = (time.monotonic_ns() - t0) / 1e6
            _log_request(request_id, elapsed_ms, state, outcome, error_class, error_msg)
            raise HTTPException(status_code=500, detail=str(e)) from e

        try:
            if np.isnan(native).any():
                raise ValueError("predictor emitted NaN")
            actions = adapter.postprocess(native)
        except (ValueError, AssertionError, NotImplementedError) as e:
            outcome = "postprocess_error"
            error_class = type(e).__name__
            error_msg = str(e)
            elapsed_ms = (time.monotonic_ns() - t0) / 1e6
            _log_request(request_id, elapsed_ms, state, outcome, error_class, error_msg)
            raise HTTPException(status_code=500, detail=str(e)) from e

        elapsed_ms = (time.monotonic_ns() - t0) / 1e6
        _log_request(request_id, elapsed_ms, state, outcome, None, None)
        return PredictResponse(actions=actions)

    return app


def _log_request(
    request_id: str,
    elapsed_ms: float,
    state: dict,
    outcome: str,
    error_class: str | None,
    error_msg: str | None,
) -> None:
    payload = {
        "ts_ns": time.monotonic_ns(),
        "request_id": request_id,
        "elapsed_ms": round(elapsed_ms, 3),
        "predictor": state["predictor_class"],
        "outcome": outcome,
    }
    if elapsed_ms > _LATENCY_BUDGET_MS:
        payload["latency_budget_ms"] = _LATENCY_BUDGET_MS
        payload["latency_budget_exceeded"] = True
    if error_class:
        payload["error_class"] = error_class
        payload["error_msg"] = error_msg
    logger.warning(json.dumps(payload)) if outcome != "ok" else logger.info(json.dumps(payload))
```

```python
# src/vla_project/deployment/__init__.py
"""HTTP inference server for X-VLA-Adapter checkpoints. See spec at
docs/superpowers/specs/2026-05-06-vla-inference-server-design.md."""

from vla_project.deployment.inference_server import build_app

__all__ = ["build_app"]
```

- [ ] **Step 3: codex review (minimal TDD test in Step 1; full smoke lands in Task 11)**

```bash
codex exec -m gpt-5.5 --skip-git-repo-check 'Review src/vla_project/deployment/inference_server.py — build_app + /predict + /healthz + per-request log emission. Implements spec §Section 6 failure handling table + observability fields + latency_budget_exceeded log marker. Concerns: (a) does HTTP code mapping match spec lines 530-540 (422 for preprocess / invalid request, 500 for predictor / postprocess errors INCLUDING the Phase 0 NotImplementedError stub)? (b) is the latency log emitted on success AND failure paths? (c) does inject_sleep_s correctly path to /predict only (not /healthz)? Reply terse.'
```

- [ ] **Step 4: Run minimal test → pass**

```bash
uv run pytest tests/test_inference_server_minimal.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/vla_project/deployment/inference_server.py src/vla_project/deployment/__init__.py tests/test_inference_server_minimal.py
git commit -m "$(cat <<'EOF'
feat(deployment): build_app + /predict + /healthz + latency log

FastAPI app factory implementing spec §Section 6 failure handling
(422 for invalid request, 500 for predictor / postprocess errors including
NotImplementedError stub), observability JSON log line per request, and
slow-path latency_budget_exceeded marker. /healthz returns predictor
class + ready timestamp.

inject_sleep_s wires test-only --inject-sleep flag from scripts/serve.py.

End-to-end behavior is exercised by tests/test_serve_smoke.py (Task 11);
this commit lands a minimal /healthz smoke for TDD discipline.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 9: scripts/serve.py

**Files:**
- Create: `scripts/serve.py`

- [ ] **Step 1: Implement** (per spec §Section 2 lines 109-136):

```python
# scripts/serve.py
"""Entry point for the inference HTTP server.

Run:
  uv run python scripts/serve.py \
    --predictor hold_position \
    --deploy-config configs/deploy/v36_libero_spatial.yaml \
    --domain-id 0 \
    --port 8001

For xvla_adapter mode (Phase 1):
  uv run python scripts/serve.py \
    --predictor xvla_adapter \
    --checkpoint /path/to/v36_export \
    --deploy-config configs/deploy/v36_libero_spatial.yaml \
    --domain-id 0 \
    --port 8001
"""
from __future__ import annotations

import argparse
import logging
import sys

import uvicorn

from vla_project.deployment.inference_server import build_app


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="X-VLA-Adapter inference HTTP server")
    ap.add_argument("--predictor", choices=["hold_position", "xvla_adapter"], required=True)
    ap.add_argument("--checkpoint", required=False, default=None,
                    help="ckpt export dir (required iff --predictor xvla_adapter)")
    ap.add_argument("--deploy-config", required=True)
    ap.add_argument("--domain-id", type=int, required=True)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--inject-sleep", type=float, default=0.0,
                    help="test-only: sleep N seconds before predict to exercise the latency log path")
    args = ap.parse_args(argv)

    if args.predictor == "xvla_adapter" and args.checkpoint is None:
        ap.error("--checkpoint required when --predictor xvla_adapter")

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    app = build_app(
        predictor_kind=args.predictor,
        checkpoint=args.checkpoint,
        deploy_config_path=args.deploy_config,
        domain_id=args.domain_id,
        inject_sleep_s=args.inject_sleep,
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: codex + commit**

```bash
codex exec -m gpt-5.5 --skip-git-repo-check 'Review scripts/serve.py. Spec §Section 2 lines 109-136, must remain thin (CLAUDE.md "Scripts" rule). Concerns: any logic outside argparse + build_app? Reply terse.'
git add scripts/serve.py
git commit -m "feat(deployment): scripts/serve.py argparse entry

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 10: Deploy yamls + MimicRec pairing example

**Files:**
- Create: `configs/deploy/_template.yaml`
- Create: `configs/deploy/v36_libero_spatial.yaml`
- Create: `configs/deploy/mimicrec_pairing_example.yaml`

- [ ] **Step 1: Write `_template.yaml`** (commented, copy-paste-edit; full content from spec §Section 4 lines 263-353)

Use the spec's full template verbatim, with all comments preserved. (The plan does not duplicate the YAML here — see spec for the canonical text.) Save to `configs/deploy/_template.yaml`.

- [ ] **Step 2: Write `v36_libero_spatial.yaml`** (concrete, no comments needed — refers to v36 trained on `libero_spatial_no_noops`):

```yaml
# configs/deploy/v36_libero_spatial.yaml
# Pairs with: training configs/train/libero_spatial_v36.yaml + a MimicRec
# contract YAML with response.action.normalization.method=none
# (see configs/deploy/mimicrec_pairing_example.yaml for one such contract).
ckpt:
  expected_unnorm_key: libero_spatial_no_noops
  expected_action_chunk_len: 8
  expected_action_dim: 7
  expected_proprio_dim: 8
request_fields:
  scene_image: image_primary
  wrist_image: image_wrist
  proprio: proprio
  instruction: instruction
proprio:
  source:
    components:
      - { name: joint_pos,   dims: 6, units: deg }
      - { name: gripper_pos, dims: 1, units: normalized_neg1_pos1 }
    total_dim: 7
  adapt:
    steps:
      - { op: deg_to_rad, source: joint_pos, dims: 6 }
      - { op: copy,       source: gripper_pos, dims: 1 }
      - { op: pad_zeros,  dims: 1 }
    output_dim: 8
  normalization:
    method: q99
    stats_key: proprio
action:
  native:
    units: meter_axisangle_rad
    frame: world
    gripper:
      kind: absolute
      units: normalized_0_1
      sign: { closed: 0, open: 1 }
  contract:
    units: meter_axisangle_rad
    frame: ee_local
    gripper:
      kind: absolute
      units: normalized_0_1
      sign: { closed: 0, open: 1 }
  denormalization:
    method: q99
    stats_key: action
  frame_conversion:
    method: none
holdposition:
  gripper_native_midpoint: 0.5
wire_only_smoke: true   # v36 native frame=world ≠ contract.frame=ee_local; this server is for
                        # smoke testing only until a contract-frame-trained ckpt exists.
runtime:
  device: cuda:0
  dtype: bf16
  torch_compile: off
  warmup_iters: 1
```

- [ ] **Step 3: Write `mimicrec_pairing_example.yaml`** — informational copy of the contract YAML the operator should drop into MimicRec:

```yaml
# configs/deploy/mimicrec_pairing_example.yaml
#
# This file is INFORMATIONAL — it shows the MimicRec-side contract YAML that
# pairs with this server's v36_libero_spatial.yaml deploy. It does NOT live
# in this server's process; copy it (or its key fields) into MimicRec's
# configs/inference/<your-name>.yaml.
#
# Critical fields:
#   - response.action.normalization.method = none  (server emits physical units)
#   - response.chunk.expected_size = 8             (matches v36 action_chunk_len)
#   - response.action.gripper.units = normalized_0_1
#   - response.action.frame = ee_local             (server has wire_only_smoke=true)
name: x_vla_v36_libero_spatial_smoke
description: "X-VLA-Adapter v36 (LIBERO Spatial single-domain), HoldPosition smoke."
endpoint:
  url: "http://localhost:8001/predict"
  method: POST
  timeout_s: 5.0
  retry: { max_attempts: 0 }
request:
  images:
    front: { field: image_primary, encoding: jpeg_base64, resize: [224, 224], jpeg_quality: 90 }
    wrist: { field: image_wrist,   encoding: jpeg_base64, resize: [224, 224], jpeg_quality: 90 }
  state:
    field: proprio
    components: [joint_pos, gripper_pos]
    normalization: { method: none }
  instruction:
    field: instruction
  extra_fields:
    model_version: x_vla_v36_libero_spatial
response:
  actions_path: actions
  chunk:
    expected_size: 8
    on_size_mismatch: use_actual
  action:
    type: ee_delta
    frame: ee_local
    pose:
      units: meter_axisangle_rad
    gripper:
      kind: absolute
      units: normalized_0_1
    components: [ee_delta, gripper]
    normalization: { method: none }
```

- [ ] **Step 4: codex review on all three yaml files**

```bash
codex exec -m gpt-5.5 --skip-git-repo-check 'Review configs/deploy/{_template,v36_libero_spatial,mimicrec_pairing_example}.yaml. v36 deploy yaml must satisfy DeployConfig pydantic schema in src/vla_project/deployment/domain_adapter.py. Pairing example must have response.action.normalization.method=none and chunk.expected_size=8. Concerns: any field in v36 deploy yaml that would fail validate_startup_hold_position? Any inconsistency between v36 deploy yaml and pairing example? Reply terse.'
```

- [ ] **Step 5: Human review (gate)**

Spec §Section 6 Phase 0 acceptance gate item 5 requires `_template.yaml` to be
**human-reviewed** (codex review is a peer opinion, not a substitute). Pause
here and ask the user to read `configs/deploy/_template.yaml`. Do not commit
until the user explicitly approves. If the user requests changes, apply them
and re-run codex.

- [ ] **Step 6: Commit (after user approval)**

```bash
git add configs/deploy/_template.yaml configs/deploy/v36_libero_spatial.yaml configs/deploy/mimicrec_pairing_example.yaml
git commit -m "feat(deployment): deploy yamls — _template + v36_libero_spatial + mimicrec pairing example

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 11: test_serve_smoke.py — end-to-end via FastAPI TestClient

**Files:**
- Create: `tests/test_serve_smoke.py`

- [ ] **Step 1: Tests** (the canonical Phase 0 server-side smoke):

```python
# tests/test_serve_smoke.py
"""End-to-end smoke for the FastAPI app via TestClient.

Spec §Section 6 Phase 0 acceptance gate items:
  (a) valid request → 200 with shape [8, 7], cols 0..5 zero, col 6 ≈ 0.5
  (b) missing scene_image → 422
  (c) proprio length wrong → 422
  (d) missing wrist_image when soft-required (zero-fill) → 200
      (the hard-required case requires xvla_adapter mode → Phase 1)
  (e) injected sleep → 200 + latency_budget_exceeded log
  (f) /healthz → ok
"""
import base64
import io
import json
import logging

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from vla_project.deployment.inference_server import build_app

DEPLOY_YAML = "configs/deploy/v36_libero_spatial.yaml"


def _b64_jpeg(size=224):
    img = Image.new("RGB", (size, size), color=(127, 127, 127))
    buf = io.BytesIO(); img.save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode("ascii")


@pytest.fixture
def client():
    app = build_app(
        predictor_kind="hold_position",
        checkpoint=None,
        deploy_config_path=DEPLOY_YAML,
        domain_id=0,
        inject_sleep_s=0.0,
    )
    return TestClient(app)


@pytest.fixture
def slow_client():
    app = build_app(
        predictor_kind="hold_position",
        checkpoint=None,
        deploy_config_path=DEPLOY_YAML,
        domain_id=0,
        inject_sleep_s=0.4,
    )
    return TestClient(app)


def _valid_request_body():
    return {
        "image_primary": _b64_jpeg(),
        "image_wrist": _b64_jpeg(),
        "proprio": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 0.5],
        "instruction": "pick up the bottle",
    }


def test_healthz_returns_ok(client):
    r = client.get("/healthz")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["predictor"] == "HoldPositionChunkPredictor"
    assert "ready_at_ns" in body


def test_predict_holdposition_shape_and_content(client):
    r = client.post("/predict", json=_valid_request_body())
    assert r.status_code == 200
    body = r.json()
    actions = body["actions"]
    assert len(actions) == 8
    assert all(len(row) == 7 for row in actions)
    for row in actions:
        for v in row[:6]:
            assert v == pytest.approx(0.0)
        assert row[6] == pytest.approx(0.5)


def test_predict_missing_scene_image_is_422(client):
    body = _valid_request_body()
    del body["image_primary"]
    r = client.post("/predict", json=body)
    assert r.status_code == 422


def test_predict_proprio_wrong_length_is_422(client):
    body = _valid_request_body()
    body["proprio"] = [0.0] * 5  # too short
    r = client.post("/predict", json=body)
    assert r.status_code == 422


def test_predict_wrist_omitted_zero_fills_and_returns_200(client):
    """v36 has wrist_view_dropout_p=0.3 → soft-required. Without ckpt loaded
    (HoldPosition mode), validate_startup_hold_position skips the wrist
    requirement check entirely. The runtime path zero-fills missing wrist."""
    body = _valid_request_body()
    body.pop("image_wrist")
    r = client.post("/predict", json=body)
    assert r.status_code == 200


def test_predict_hard_required_wrist_missing_at_request_returns_422(tmp_path):
    """When the deploy yaml's request_fields.wrist_image is set AND the ckpt
    cfg flags hard-required wrist, runtime should reject a request that
    omits wrist with 422. We can't load a real ckpt in Phase 0, so we
    simulate by forcing the `inference_server.build_app` to set a server-
    side flag `_wrist_required=True` via an xvla_adapter mode startup with a
    synthetic meta.json fixture."""
    import json
    # Build a synthetic ckpt dir that satisfies validate_startup_xvla AND
    # has use_wrist_bridge=True so wrist is hard-required.
    ckpt_dir = tmp_path / "fake_v33"
    ckpt_dir.mkdir()
    meta = {
        "step": 0,
        "cfg": {
            "model": {
                "num_domains": 1,
                "use_wrist_bridge": True,
                "wrist_in_llm": False,
                "wrist_view_dropout_p": 0.0,
            },
            "data": {"unnorm_key": "libero_spatial_no_noops", "action_chunk_len": 8},
        },
        "norm_stats": {
            "libero_spatial_no_noops": {
                "action": {"mean": [0.0]*7, "std": [1.0]*7, "q01": [-1.0]*7, "q99": [1.0]*7,
                           "mask": [True]*6 + [False]},
                "proprio": {"mean": [0.0]*8, "std": [1.0]*8, "q01": [-1.0]*8, "q99": [1.0]*8,
                            "mask": [True]*8},
            }
        },
    }
    (ckpt_dir / "meta.json").write_text(json.dumps(meta))
    app = build_app(
        predictor_kind="xvla_adapter",
        checkpoint=ckpt_dir,
        deploy_config_path=DEPLOY_YAML,
        domain_id=0,
        inject_sleep_s=0.0,
    )
    c = TestClient(app)
    body = _valid_request_body()
    body.pop("image_wrist")
    r = c.post("/predict", json=body)
    # Phase 0: predict() raises NotImplementedError → 500; before it does,
    # preprocess+request_fields validation should reject hard-required wrist
    # absent with 422.
    assert r.status_code == 422
    assert "wrist" in r.json()["detail"].lower()


def test_inject_sleep_emits_latency_budget_exceeded(slow_client, caplog):
    """Server still returns 200 within MimicRec's 5s timeout, but the log
    line carries latency_budget_exceeded=true. Phase 0 acceptance gate item 4."""
    caplog.set_level(logging.INFO, logger="vla_project.deployment")
    r = slow_client.post("/predict", json=_valid_request_body())
    assert r.status_code == 200
    # Find the per-request log line emitted by inference_server._log_request.
    matched = [rec for rec in caplog.records if rec.name == "vla_project.deployment"]
    assert matched, "expected at least one log record from vla_project.deployment"
    payload = json.loads(matched[-1].getMessage())
    assert payload.get("latency_budget_exceeded") is True
    assert payload["elapsed_ms"] > 266.0
```

- [ ] **Step 2: Run → many fail at first (refining test_predict_wrist_omitted may need adapter behavior tweaks)**

```bash
uv run pytest tests/test_serve_smoke.py -v
```

If `test_predict_wrist_omitted_zero_fills_and_returns_200` fails because preprocess raises on missing wrist with `wrist_was_provided=False`, that means DomainAdapter is rejecting the missing field — which is correct for hard-required ckpts but we are in HoldPosition mode where wrist is unused. Fix: either preprocess unconditionally allows None wrist (current behavior — it already zero-fills), OR add a `predictor_uses_wrist` flag to DomainAdapter. Current `domain_adapter.preprocess` zero-fills when None, so the test should pass.

- [ ] **Step 3: codex review**

```bash
codex exec -m gpt-5.5 --skip-git-repo-check 'Review tests/test_serve_smoke.py — Phase 0 acceptance gate items (a)-(f). Concerns: (a) does the wrist-omitted test correctly exercise zero-fill in HoldPosition mode? (b) is the slow-path log capture correct (caplog level + logger name)? (c) any case missed from spec §Section 6 Phase 0 acceptance gate items 1-6? Reply terse.'
```

- [ ] **Step 4: Commit**

```bash
git add tests/test_serve_smoke.py
git commit -m "$(cat <<'EOF'
test(deployment): test_serve_smoke.py end-to-end via FastAPI TestClient

Covers Phase 0 acceptance gate items (a)-(f): valid request → [8, 7]
zeros + 0.5 gripper, malformed → 422, wrist omitted → zero-filled 200,
slow-path → latency_budget_exceeded log, /healthz.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 12: deployment/README.md

**Files:**
- Create: `src/vla_project/deployment/README.md`

- [ ] **Step 1: Write README** (operator-facing; covers spec §Section 6 acceptance gate item 6):

```markdown
# `vla_project.deployment` — VLA Inference HTTP Server

Hosts an X-VLA-Adapter checkpoint behind MimicRec's `POST /predict` contract.

**Design:** see `docs/superpowers/specs/2026-05-06-vla-inference-server-design.md`.

---

## Phase status

- **Phase 0 (this code):** Skeleton + `HoldPositionChunkPredictor` end-to-end. `XVLAAdapterChunkPredictor` is a stub raising `NotImplementedError("Phase 1")`. `ModelRuntime.__call__` likewise.
- **Phase 1 (next):** Train v36 ckpt, plug `XVLAAdapterChunkPredictor` real forward path + `ModelRuntime` torch wrapper. Live latency benchmark.

---

## Starting the server

### HoldPosition mode (Phase 0; no GPU / no ckpt required)

```bash
uv run python scripts/serve.py \
  --predictor hold_position \
  --deploy-config configs/deploy/v36_libero_spatial.yaml \
  --domain-id 0 \
  --port 8001
```

Verify:
```bash
curl http://localhost:8001/healthz
# {"status":"ok","predictor":"HoldPositionChunkPredictor","ready_at_ns":...}
```

### XVLAAdapter mode

In **Phase 0**, `--predictor xvla_adapter` works as a startup-validation
exerciser only. It loads `meta.json` (no GPU / no torch model load), runs all
`validate_startup_xvla` hard-fail asserts, and starts the FastAPI app — but
`POST /predict` returns **HTTP 500** with `error_class=NotImplementedError`
because `XVLAAdapterChunkPredictor.predict()` is stubbed for Phase 1.

```bash
# Phase 0: meta.json load + assert path
uv run python scripts/serve.py \
  --predictor xvla_adapter \
  --checkpoint ~/X-VLA-Adapter_export/v33_step40000 \
  --deploy-config configs/deploy/v36_libero_spatial.yaml \
  --domain-id 0 \
  --port 8001
# /healthz → ok; /predict → 500 NotImplementedError
```

In **Phase 1**, the same command (with a v36 ckpt) will produce real action
chunks. GPU is required only in Phase 1.

---

## Swapping checkpoints

There is no reload endpoint. Kill the process and restart with the new
`--checkpoint` value. systemd / docker restart policies handle the gap.

---

## Adding a new robot deploy yaml

1. Copy `configs/deploy/_template.yaml` to `configs/deploy/<robot>_<model>.yaml`.
2. Fill in the four `ckpt.expected_*` fields from the ckpt's `meta.json` cfg.
3. Set `request_fields` to the canonical wire names (`image_primary`, `image_wrist`, `proprio`, `instruction`). The MimicRec contract YAML on the client side **must** use these canonical names in Phase 0; pydantic alias remap for arbitrary contract names is Phase 1 work.
4. Set `proprio.{source, adapt}` to match the robot's raw proprio layout (see template comments).
5. Set `action.{native, contract}` per the ckpt's training output convention and the MimicRec contract's response convention. If `native.frame != contract.frame` and you have not implemented `frame_conversion`, set `wire_only_smoke: true` to bypass the startup assertion (motion will not be physically correct).
6. (Optional) Adjust `holdposition.gripper_native_midpoint` if the native gripper convention is not `normalized_0_1`.

Then start the server pointing at the new yaml.

---

## Known limitations (Phase 0)

- **Frame conversion not implemented.** Set `wire_only_smoke: true` for cross-frame deploy yamls (motion will be wrong; useful only for wire-format smoke).
- **HoldPosition is not a safety fallback.** It is for wire-shape smoke / pre-model-trained sentinel. MimicRec's slow-stop ramp is the real fallback for missing-action conditions.
- **No `/admin/reload` endpoint.** Kill + restart for ckpt swap.
- **`max_inflight=1` only** (matches MimicRec's MVP setting). No request batching.
- **Latency benchmark deferred.** Target p95 < 266 ms after `torch.compile` warmup; not measured in Phase 0.

---

## Phase 0 acceptance verification

Run the named test files:

```bash
uv run pytest \
  tests/test_deployment_schemas.py \
  tests/test_domain_adapter.py \
  tests/test_predictor_holdposition.py \
  tests/test_predictor_xvla_adapter.py \
  tests/test_runtime_load.py \
  tests/test_serve_smoke.py \
  -q
```

For the MimicRec integration smoke (Phase 0 acceptance gate item 3):

```bash
# Terminal 1: start the server
uv run python scripts/serve.py \
  --predictor hold_position \
  --deploy-config configs/deploy/v36_libero_spatial.yaml \
  --domain-id 0 --port 8001

# Terminal 2: run MimicRec smoke (after copying configs/deploy/mimicrec_pairing_example.yaml
# into MimicRec's configs/inference/ directory and editing endpoint.url to
# http://localhost:8001/predict)
cd /home/takakimaeda/MimicRec
.venv/bin/python scripts/smoke_inference_real_data.py
```

Expected output:
```
✅ inference mock pipeline works end-to-end with real data
IK failures: 0/N    # zero ee_delta → no IK displacement
```
```

- [ ] **Step 2: codex review**

```bash
codex exec -m gpt-5.5 --skip-git-repo-check 'Review src/vla_project/deployment/README.md. Spec §Section 6 Phase 0 acceptance gate item 6 lists required content (start command, ckpt swap, new robot yaml, known limitations). Concerns: any required item missing? Any contradiction with spec or other code? Reply terse.'
```

- [ ] **Step 3: Commit**

```bash
git add src/vla_project/deployment/README.md
git commit -m "docs(deployment): operator-facing README for the inference server

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 13: Phase 0 acceptance gate verification (manual)

This is not a code task — it is a manual gate. Run all six named test files and the MimicRec smoke.

- [ ] **Step 1: Run named test pyramid**

```bash
uv run pytest \
  tests/test_deployment_schemas.py \
  tests/test_domain_adapter.py \
  tests/test_predictor_holdposition.py \
  tests/test_predictor_xvla_adapter.py \
  tests/test_runtime_load.py \
  tests/test_serve_smoke.py \
  -q
```

Expected: ALL PASS, no warnings escalated to errors.

- [ ] **Step 2: Boot server in HoldPosition mode + verify /healthz within 30 s**

```bash
uv run python scripts/serve.py \
  --predictor hold_position \
  --deploy-config configs/deploy/v36_libero_spatial.yaml \
  --domain-id 0 --port 8001 &
SERVER_PID=$!
sleep 5
curl -sS http://localhost:8001/healthz
# expected: {"status":"ok","predictor":"HoldPositionChunkPredictor",...}
kill $SERVER_PID
```

- [ ] **Step 3: Run MimicRec smoke against running server**

```bash
# Server running as in Step 2.
cp configs/deploy/mimicrec_pairing_example.yaml \
   /home/takakimaeda/MimicRec/configs/inference/x_vla_v36_smoke.yaml
# (Edit the copy to set endpoint.url=http://localhost:8001/predict if needed.)
cd /home/takakimaeda/MimicRec
.venv/bin/python scripts/smoke_inference_real_data.py 2>&1 | tee /tmp/mimicrec_smoke.log
```

Expected `/tmp/mimicrec_smoke.log` contains:
- `✅ inference mock pipeline works end-to-end with real data`
- `IK failures: 0/N`

- [ ] **Step 3b: Independent response-shape verification (spec acceptance gate item 3)**

```bash
# Server still running from Step 2.
PAYLOAD=$(python - <<'PY'
import base64, io, json
from PIL import Image
buf = io.BytesIO(); Image.new("RGB", (224, 224), (127, 127, 127)).save(buf, format="JPEG")
b64 = base64.b64encode(buf.getvalue()).decode()
print(json.dumps({
  "image_primary": b64, "image_wrist": b64,
  "proprio": [0,0,0,0,0,0,0], "instruction": "pick up the bottle",
}))
PY
)
RESP=$(curl -sS -X POST http://localhost:8001/predict \
  -H 'Content-Type: application/json' -d "$PAYLOAD")
echo "$RESP" | python - <<'PY'
import json, sys
r = json.loads(sys.stdin.read())
a = r["actions"]
assert len(a) == 8, f"expected 8 rows, got {len(a)}"
for row in a:
    assert len(row) == 7, f"expected 7 cols, got {len(row)}"
    assert all(v == 0.0 for v in row[:6]), f"ee_delta cols not zero: {row[:6]}"
    assert abs(row[6] - 0.5) < 1e-6, f"gripper not midpoint: {row[6]}"
print("✓ response shape [8, 7], ee_delta zeros, gripper midpoint 0.5")
PY
```

Expected: `✓ response shape [8, 7], ee_delta zeros, gripper midpoint 0.5`.

- [ ] **Step 4: Slow-path test**

```bash
uv run python scripts/serve.py \
  --predictor hold_position \
  --deploy-config configs/deploy/v36_libero_spatial.yaml \
  --domain-id 0 --port 8001 \
  --inject-sleep 0.4 &
SERVER_PID=$!
sleep 5
curl -sS -X POST http://localhost:8001/predict \
  -H "Content-Type: application/json" \
  -d "{\"image_primary\":\"$(python -c 'from PIL import Image; import io, base64; img=Image.new(\"RGB\",(224,224)); buf=io.BytesIO(); img.save(buf,format=\"JPEG\"); print(base64.b64encode(buf.getvalue()).decode())')\",\"proprio\":[0,0,0,0,0,0,0],\"instruction\":\"x\"}"
# expected: 200 OK with actions array; server stdout shows latency_budget_exceeded:true log
kill $SERVER_PID
```

- [ ] **Step 5: Final codex review of the whole branch**

```bash
codex exec -m gpt-5.5 --skip-git-repo-check 'Review the whole feature branch (use git diff main..HEAD if available, else --base main). Phase 0 of the VLA inference HTTP server per docs/superpowers/specs/2026-05-06-vla-inference-server-design.md. Concerns: any spec requirement still uncovered? Any test that does not actually exercise the spec-required behavior? Reply terse, with line numbers.'
```

- [ ] **Step 6: All gate items met → mark Phase 0 complete**

If any item fails, file a follow-up task and do not declare Phase 0 done.

---

## Self-review checklist (run after writing this plan, before commit)

**Spec coverage:**
- §Section 1 Overview → Tasks 1, 8 (build_app), 12 (README startup commands)
- §Section 2 Module boundaries → Tasks 1-9 (file-by-file)
- §Section 3 Data flow → Tasks 5 (DomainAdapter), 8 (build_app /predict), 11 (smoke)
- §Section 4 Deploy yaml + ckpt schema → Tasks 5 (DeployConfig + validators), 6 (ModelRuntime meta load), 10 (yaml files)
- §Section 5 ChunkPredictor + ModelRuntime → Tasks 3 (ABC), 4 (HoldPosition), 6 (ModelRuntime stub), 7 (XVLAAdapter stub)
- §Section 6 Failure / observability / testing / Phase 0 acceptance → Tasks 8 (logging), 11 (test_serve_smoke), 12 (README), 13 (manual gate)

**Placeholder scan:** none.

**Type consistency:** ChunkPredictor / DomainAdapter / DeployConfig / build_app names match across all tasks. ModelRuntime.from_export classmethod, signature consistent. XVLAAdapterChunkPredictor ctor args verbatim from spec line 466.

**Implementation order dependencies (numbered as written; execute in this order):**
- Task 1 (deps) → all
- Task 2 (schemas) → 5 (DomainAdapter), 8 (build_app), 11 (smoke)
- Task 3 (ABC) → 4 (HoldPosition), 7 (XVLA stub)
- Task 4 (HoldPosition) → 8 (build_app branches)
- Task 5 (DomainAdapter) → 8 (build_app), 11 (smoke)
- Task 6 (ModelRuntime stub) → 8 (build_app branches)
- Task 7 (XVLA stub) → 8 (build_app branches)
- **Task 10 (deploy yamls) MUST come before Task 8** because Task 8's minimal TDD test loads `configs/deploy/v36_libero_spatial.yaml`. Recommended execution order: 1 → 2 → 3 → 4 → 5 → 6 → 7 → 10 → 8 → 9 → 11 → 12 → 13.
- Task 8 (build_app) → 9 (serve.py), 11 (smoke)
- Task 11 (smoke tests) → 13 (manual gate)
- Task 12 (README) → 13 (manual gate)
