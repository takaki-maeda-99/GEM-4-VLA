# Inference Server Request Validation (Phase 0 add-on) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add 5 request-validation features (F1 image sanity, F2 instruction bytes, F3 proprio OOD, F4 wire-typo guard, F5 `/admin/schema`) to the Phase 0 inference server, surfacing "obvious mistake" errors at the request boundary without changing what the model sees.

**Architecture:** Layered TDD. Schema-layer validators in `schemas.py` (F2 byte length, F4 typo guard via Damerau-Levenshtein). Preprocess-layer checks in `domain_adapter.py` (F1 image header sanity, F3 proprio NaN/inf + OOD warn/hard-reject). New GET `/admin/schema` route in `inference_server.py` (F5). All thresholds are module-level constants near their use site; no new deploy yaml fields; no tokenizer load (deferred to Phase 1).

**Tech Stack:** pydantic v2, FastAPI, numpy, PIL, Python stdlib (no new deps).

**Spec:** `docs/superpowers/specs/2026-05-08-server-request-validation-design.md`

---

## File structure

**Modified files:**

| Path | Responsibility after change |
|---|---|
| `src/vla_project/deployment/schemas.py` | Wire schema + F2 byte-length validator + F4 typo guard model_validator. Adds module-level `INSTRUCTION_MAX_BYTES`. |
| `src/vla_project/deployment/domain_adapter.py` | Preprocess + startup validation, with F1 (header-first image bound in `_decode_jpeg_b64`), F3a (NaN/inf reject in `preprocess`), F3b (OOD warn/raise in `_normalize_proprio`). Adds `IMAGE_MIN_SIDE`, `IMAGE_MAX_SIDE`, `PROPRIO_OOD_WARN_ABS`, `PROPRIO_OOD_HARD_ABS`. |
| `src/vla_project/deployment/inference_server.py` | FastAPI app factory + routes; new GET `/admin/schema` route. |
| `src/vla_project/deployment/README.md` | Append 5 new test files to the Phase 0 acceptance gate command. |
| `tests/test_domain_adapter.py` | Extend with 1 NaN proprio case + 1 image-bound direct-call case. |
| `tests/test_serve_smoke.py` | Extend with 1 typo case (`image_pirmary`) for end-to-end FastAPI coverage. |
| `tests/test_inference_server_minimal.py` | Extend to assert `/admin/schema` responds alongside `/healthz`. |

**New files:**

| Path | Responsibility |
|---|---|
| `tests/test_validation_image_sanity.py` | F1 unit + integration tests. |
| `tests/test_validation_prompt.py` | F2 byte-length tests (empty allowed, 10 KB cap, UTF-8 boundary). |
| `tests/test_validation_proprio.py` | F3 NaN/inf + OOD warn/hard-reject tests. |
| `tests/test_validation_typo.py` | F4 Damerau-Levenshtein near-miss + image-prefix fallback tests. |
| `tests/test_admin_schema.py` | F5 endpoint shape + content tests for both predictor modes. |

**Unchanged files (per spec):**

- `src/vla_project/deployment/predictors/{base,hold_position,xvla_adapter}.py`
- `src/vla_project/deployment/runtime.py`
- `configs/deploy/*.yaml`
- No new `prompt_processor.py`, no new `constants.py`.

---

## Task ordering rationale

Tasks are ordered to ensure each commit produces a passing test suite without depending on later work:

1. **Task 1 (F2)** — schema-only, smallest unit, no preprocess changes.
2. **Task 2 (F4)** — schema-only, builds on Task 1's schema-layer changes (independent constants but same file edit pattern).
3. **Task 3 (F1)** — touches `domain_adapter.py` only (`_decode_jpeg_b64`).
4. **Task 4 (F3)** — touches `domain_adapter.py` only (`preprocess` + `_normalize_proprio`); independent of Task 3.
5. **Task 5 (F5)** — touches `inference_server.py`; depends on F1/F2/F3 constants existing (so `/admin/schema` can read them).
6. **Task 6** — extends existing tests + updates README. Pure test plumbing, no production-code changes.
7. **Task 7** — full acceptance gate run + MimicRec smoke regression check.

---

## Task 1: F2 — Instruction byte-length validator + RequestValidationError serializer fix

**Files:**
- Modify: `src/vla_project/deployment/schemas.py`
- Modify: `src/vla_project/deployment/inference_server.py` (pre-existing serializer bug surfaces here)
- Create: `tests/test_validation_prompt.py`

> **Pre-existing bug surfaced by Task 1:** the current `RequestValidationError` exception handler in `inference_server.py` returns `exc.errors()` directly to `JSONResponse`. In pydantic v2, when a `field_validator` or `model_validator` raises `ValueError`, the error dict's `ctx` field carries the original exception object (`{"error": ValueError(...)}`), which Starlette's JSON encoder cannot serialize. No existing test triggers this because no validator in the current schema raises `ValueError`. Task 1 introduces the first such validator (F2 byte length), so the fix must land here, not later.

- [ ] **Step 1.1: Write the failing test**

Create `tests/test_validation_prompt.py`:

```python
"""F2: instruction byte-length sanity (Phase 0 — pydantic-only).

Spec: docs/superpowers/specs/2026-05-08-server-request-validation-design.md §F2.
Empty allowed (matches Phase 0 spec L153 'may be empty in pre-start states').
Upper bound 10 000 UTF-8 bytes.
"""
import base64

import pytest
from pydantic import ValidationError

from vla_project.deployment.schemas import PredictRequest


def _b64_jpeg(n_bytes: int = 64) -> str:
    return base64.b64encode(b"\xff\xd8\xff" + b"\x00" * n_bytes).decode("ascii")


def _kwargs(instruction: str) -> dict:
    return dict(
        image_primary=_b64_jpeg(),
        proprio=[0.0] * 7,
        instruction=instruction,
    )


def test_instruction_empty_string_is_valid():
    """Phase 0 spec L153: instruction may be empty in pre-start states."""
    req = PredictRequest(**_kwargs(""))
    assert req.instruction == ""


def test_instruction_short_ascii_is_valid():
    req = PredictRequest(**_kwargs("pick up the bottle"))
    assert req.instruction == "pick up the bottle"


def test_instruction_at_byte_limit_is_valid():
    """10 000 ASCII bytes is the boundary — must accept."""
    s = "a" * 10_000
    req = PredictRequest(**_kwargs(s))
    assert len(req.instruction.encode("utf-8")) == 10_000


def test_instruction_over_byte_limit_is_rejected():
    s = "a" * 10_001
    with pytest.raises(ValidationError) as exc:
        PredictRequest(**_kwargs(s))
    assert "byte length" in str(exc.value).lower()


def test_instruction_multibyte_utf8_counted_in_bytes_not_chars():
    """Japanese 'あ' = 3 bytes in UTF-8. 5000 chars × 3 = 15000 bytes > 10000."""
    s = "あ" * 5000
    with pytest.raises(ValidationError) as exc:
        PredictRequest(**_kwargs(s))
    assert "byte length" in str(exc.value).lower()


def test_instruction_just_under_byte_limit_is_valid():
    """9 999 ASCII bytes — boundary minus one."""
    s = "a" * 9_999
    req = PredictRequest(**_kwargs(s))
    assert len(req.instruction.encode("utf-8")) == 9_999
```

- [ ] **Step 1.2: Run test to verify it fails**

Run: `PYTHONPATH= uv run pytest tests/test_validation_prompt.py -v`

Expected: 4 PASS / 2 FAIL. The 2 failures are `test_instruction_over_byte_limit_is_rejected` and `test_instruction_multibyte_utf8_counted_in_bytes_not_chars` (both expect a `ValidationError` that no validator currently raises). The other 4 tests pass trivially because pydantic's `str` already accepts any string.

- [ ] **Step 1.3: Add byte-length validator to schemas.py**

In `src/vla_project/deployment/schemas.py`:

Replace:

```python
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
```

With:

```python
"""Pydantic v2 wire schemas for the inference HTTP server.

PredictRequest mirrors what MimicRec sends per its contract YAML; PredictResponse
is what the server returns. Field naming follows the MimicRec spec excerpts in
docs/superpowers/specs/2026-05-06-vla-inference-server-design.md §Section 3.

The wire field `_t_mono_ns` is exposed as `t_mono_ns` on the model because
pydantic v2 reserves leading-underscore names as private attributes; we use
`populate_by_name=True` + `Field(alias="_t_mono_ns")`.

Validation features (per docs/superpowers/specs/2026-05-08-server-request-
validation-design.md):
  - F2: instruction must be ≤ INSTRUCTION_MAX_BYTES UTF-8 bytes (empty allowed).
  - F4: typo guard added in a later step (model_validator).
"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


# F2: instruction byte-length sanity (UTF-8 byte count, NOT char count — pydantic's
# native max_length constrains chars, but multibyte UTF-8 means 10 000 chars can
# be 30 000+ bytes for Japanese / emoji).
INSTRUCTION_MAX_BYTES: int = 10_000


class PredictRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    image_primary: str
    image_wrist: str | None = None
    proprio: list[float]
    instruction: str
    model_version: str | None = None
    t_mono_ns: dict[str, Any] | None = Field(default=None, alias="_t_mono_ns")

    @field_validator("instruction")
    @classmethod
    def _instruction_byte_length(cls, v: str) -> str:
        n = len(v.encode("utf-8"))
        if n > INSTRUCTION_MAX_BYTES:
            raise ValueError(
                f"instruction byte length {n} > {INSTRUCTION_MAX_BYTES} "
                f"(UTF-8 byte count, not char count)"
            )
        return v
```

- [ ] **Step 1.4: Run schema-only test to verify it passes**

Run: `PYTHONPATH= uv run pytest tests/test_validation_prompt.py -v`

Expected: 6 PASS. (These tests construct `PredictRequest` directly via pydantic — the FastAPI HTTP serialization path is exercised separately in Tasks 3/5/6.)

- [ ] **Step 1.5: Add E2E HTTP test to surface the pre-existing serializer bug**

Append to `tests/test_validation_prompt.py`:

```python
def test_instruction_over_byte_limit_returns_422_via_http():
    """End-to-end: a >10 000-byte instruction must return HTTP 422 (NOT 500
    from a JSON-serialization crash in the RequestValidationError handler).

    This test exists specifically because pydantic v2's `exc.errors()` includes
    a `ctx: {'error': ValueError(...)}` field that Starlette's default JSON
    encoder cannot serialize. The fix in inference_server.py uses
    `fastapi.encoders.jsonable_encoder` to convert the exception to a string.
    """
    import base64
    import io

    from fastapi.testclient import TestClient
    from PIL import Image

    from vla_project.deployment.inference_server import build_app

    img = Image.new("RGB", (224, 224), color=(127, 127, 127))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=70)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")

    app = build_app(
        predictor_kind="hold_position",
        checkpoint=None,
        deploy_config_path="configs/deploy/v36_libero_spatial.yaml",
        domain_id=0,
    )
    c = TestClient(app)
    body = {
        "image_primary": b64,
        "proprio": [0.0] * 7,
        "instruction": "a" * 10_001,
    }
    resp = c.post("/predict", json=body)
    assert resp.status_code == 422
    detail_str = str(resp.json()["detail"]).lower()
    assert "byte length" in detail_str
```

- [ ] **Step 1.6: Run the E2E test to surface the bug**

Run: `PYTHONPATH= uv run pytest tests/test_validation_prompt.py::test_instruction_over_byte_limit_returns_422_via_http -v`

Expected: FAIL with `ValueError("Object of type ValueError is not JSON serializable")` (or similar Starlette JSON encoder error). The HTTP status will likely be 500, not 422, because the handler crashes mid-serialization.

- [ ] **Step 1.7: Fix the RequestValidationError handler to serialize via `jsonable_encoder`**

In `src/vla_project/deployment/inference_server.py`:

Add to imports near the top (after the existing `from fastapi.exceptions import RequestValidationError` line):

```python
from fastapi.encoders import jsonable_encoder
```

Replace the existing handler:

```python
    @app.exception_handler(RequestValidationError)
    async def _on_validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        request_id = request.headers.get("X-Request-Id") or uuid.uuid4().hex
        _log_request(
            request_id, elapsed_ms=0.0, state=state, domain_id=domain_id,
            outcome="invalid_request",
            error_class="RequestValidationError",
            error_msg=str(exc.errors()),
        )
        return JSONResponse(status_code=422, content={"detail": exc.errors()})
```

With:

```python
    @app.exception_handler(RequestValidationError)
    async def _on_validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        # Pydantic v2 errors() includes ctx: {"error": ValueError(...)} when a
        # field_validator/model_validator raises ValueError. The exception
        # object is not JSON-serializable; jsonable_encoder converts it via
        # str(). Without this, F2/F4 validator failures would 500 instead of
        # returning a clean 422.
        errors = jsonable_encoder(exc.errors())
        request_id = request.headers.get("X-Request-Id") or uuid.uuid4().hex
        _log_request(
            request_id, elapsed_ms=0.0, state=state, domain_id=domain_id,
            outcome="invalid_request",
            error_class="RequestValidationError",
            error_msg=str(errors),
        )
        return JSONResponse(status_code=422, content={"detail": errors})
```

- [ ] **Step 1.8: Run all tests to verify both fix and validator work**

Run: `PYTHONPATH= uv run pytest tests/test_validation_prompt.py tests/test_deployment_schemas.py tests/test_serve_smoke.py tests/test_inference_server_minimal.py -v`

Expected: All pass (7 in test_validation_prompt.py — including the new E2E one — plus all existing schema/smoke/minimal tests).

- [ ] **Step 1.9: Commit**

```bash
git add src/vla_project/deployment/schemas.py src/vla_project/deployment/inference_server.py tests/test_validation_prompt.py
git commit -m "$(cat <<'EOF'
feat(deployment): F2 instruction byte-length validator + 422 serializer fix

Add INSTRUCTION_MAX_BYTES=10000 module constant and a field_validator on
PredictRequest.instruction that rejects strings exceeding the UTF-8 byte
count. Empty strings remain valid per Phase 0 spec L153.

Also fix a latent bug in the RequestValidationError handler: pydantic v2
errors() embeds the source ValueError in ctx, which Starlette cannot
JSON-serialize. Route through fastapi.encoders.jsonable_encoder so the
422 response stays clean. This was previously latent because no schema
validator raised ValueError; F2 surfaces it.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: F4 — Wire-typo guard with Damerau-Levenshtein

**Files:**
- Modify: `src/vla_project/deployment/schemas.py`
- Create: `tests/test_validation_typo.py`

- [ ] **Step 2.1: Write the failing test**

Create `tests/test_validation_typo.py`:

```python
"""F4: wire-typo guard preserving extra='ignore' forward-compat.

Spec: docs/superpowers/specs/2026-05-08-server-request-validation-design.md §F4.
Two-rule guard:
  1. Damerau-Levenshtein distance ≤ 1 from a modeled field name → 'did you mean X?'
  2. image_* prefix that didn't trip rule 1 → 'unknown image field; known: ...'
  Otherwise: silently ignored (preserves extra='ignore' for future _trace etc.).
"""
import base64

import pytest
from pydantic import ValidationError

from vla_project.deployment.schemas import PredictRequest


def _b64_jpeg(n_bytes: int = 64) -> str:
    return base64.b64encode(b"\xff\xd8\xff" + b"\x00" * n_bytes).decode("ascii")


def _base_kwargs() -> dict:
    return dict(
        image_primary=_b64_jpeg(),
        proprio=[0.0] * 7,
        instruction="test",
    )


# ----- Rule 1: near-miss (Damerau-Levenshtein ≤ 1) -----

def test_typo_image_pirmary_caught_as_near_miss():
    """Transposition: image_pirmary (swap r-i) → distance 1 from image_primary."""
    raw = _base_kwargs()
    raw["image_pirmary"] = _b64_jpeg()
    with pytest.raises(ValidationError) as exc:
        PredictRequest.model_validate(raw)
    msg = str(exc.value)
    assert "image_pirmary" in msg
    assert "image_primary" in msg
    assert "did you mean" in msg.lower()


def test_typo_image_wirst_caught_as_near_miss():
    """Transposition: image_wirst (swap r-i) → distance 1 from image_wrist."""
    raw = _base_kwargs()
    raw["image_wirst"] = _b64_jpeg()
    with pytest.raises(ValidationError) as exc:
        PredictRequest.model_validate(raw)
    msg = str(exc.value)
    assert "image_wirst" in msg
    assert "image_wrist" in msg


def test_typo_propio_caught_as_near_miss():
    """Deletion: propio (drop r) → distance 1 from proprio."""
    raw = _base_kwargs()
    raw["propio"] = [0.0] * 7
    with pytest.raises(ValidationError) as exc:
        PredictRequest.model_validate(raw)
    msg = str(exc.value)
    assert "propio" in msg
    assert "proprio" in msg


def test_typo_t_mono_n_caught_as_near_miss():
    """Deletion: _t_mono_n (drop trailing s) → distance 1 from _t_mono_ns."""
    raw = _base_kwargs()
    raw["_t_mono_n"] = {"state": 1}
    with pytest.raises(ValidationError) as exc:
        PredictRequest.model_validate(raw)
    msg = str(exc.value)
    assert "_t_mono_n" in msg
    assert "_t_mono_ns" in msg


def test_typo_model_versionn_caught_as_near_miss():
    """Insertion: model_versionn (extra n) → distance 1 from model_version."""
    raw = _base_kwargs()
    raw["model_versionn"] = "x_vla_v36"
    with pytest.raises(ValidationError) as exc:
        PredictRequest.model_validate(raw)
    msg = str(exc.value)
    assert "model_versionn" in msg
    assert "model_version" in msg


# ----- Rule 2: image_* prefix without near miss -----

def test_image_camera_left_caught_by_image_prefix_rule():
    """image_camera_left is not within distance 1 of any modeled field; falls
    through to the image_* prefix rule."""
    raw = _base_kwargs()
    raw["image_camera_left"] = _b64_jpeg()
    with pytest.raises(ValidationError) as exc:
        PredictRequest.model_validate(raw)
    msg = str(exc.value)
    assert "image_camera_left" in msg
    assert "unknown image field" in msg.lower()
    assert "image_primary" in msg
    assert "image_wrist" in msg


# ----- Forward-compat: silently-ignored unknown fields -----

def test_underscore_request_id_silently_ignored():
    """Forward-compat: future MimicRec observability fields (_request_id, _trace,
    _session_token) must pass through. Distance from any modeled field > 1."""
    raw = _base_kwargs()
    raw["_request_id"] = "abc-123"
    req = PredictRequest.model_validate(raw)
    assert req.image_primary  # accepted


def test_underscore_trace_silently_ignored():
    raw = _base_kwargs()
    raw["_trace"] = {"span": "x"}
    req = PredictRequest.model_validate(raw)
    assert req.image_primary


def test_unrelated_unknown_field_silently_ignored():
    """A field with no edit-distance and no image_ prefix passes silently."""
    raw = _base_kwargs()
    raw["unrelated_metadata_field"] = 42
    req = PredictRequest.model_validate(raw)
    assert req.image_primary
```

- [ ] **Step 2.2: Run test to verify it fails**

Run: `PYTHONPATH= uv run pytest tests/test_validation_typo.py -v`

Expected: All 6 typo-rejection tests FAIL (no validator exists yet — extra="ignore" silently drops them). The 3 forward-compat tests PASS (already silently dropped).

- [ ] **Step 2.3: Add Damerau-Levenshtein helper + model_validator to schemas.py**

> **Note:** the "Replace … With …" blocks below apply to the **post-Task-1** state of `schemas.py` (after the `field_validator("instruction")` is already added and `INSTRUCTION_MAX_BYTES` constant is in place). If Task 1's changes are not present, apply Task 1 first.

In `src/vla_project/deployment/schemas.py`, add (above the `PredictRequest` class):

```python
# F4: typo guard — fields the wire schema models. Used by the model_validator
# below to compute Damerau-Levenshtein distance for near-miss detection.
_MODELED_FIELDS: frozenset[str] = frozenset({
    "image_primary", "image_wrist",
    "proprio", "instruction",
    "model_version", "_t_mono_ns",
})


def _damerau_levenshtein_within_one(a: str, b: str) -> bool:
    """Return True iff Damerau-Levenshtein distance(a, b) ≤ 1.

    Bounded check (we only care about distance ≤ 1), so this avoids the
    full DP matrix. Catches:
      - 0 edits (a == b)
      - 1 substitution
      - 1 insertion
      - 1 deletion
      - 1 transposition of adjacent characters (this is the Damerau extension
        — typical typos like 'pirmary' vs 'primary' are transpositions, which
        plain Levenshtein counts as distance 2.)
    """
    if a == b:
        return True
    la, lb = len(a), len(b)
    if abs(la - lb) > 1:
        return False
    # Substitution / transposition (la == lb)
    if la == lb:
        diffs = [i for i in range(la) if a[i] != b[i]]
        if len(diffs) == 1:
            return True  # single substitution
        if len(diffs) == 2 and diffs[0] + 1 == diffs[1]:
            i, j = diffs
            if a[i] == b[j] and a[j] == b[i]:
                return True  # adjacent transposition
        return False
    # Insertion / deletion: pin the longer string as `lo` (long), shorter as `sh`.
    lo, sh = (a, b) if la > lb else (b, a)
    # Try to find a single skip in `lo` that makes them equal.
    i = j = 0
    skipped = False
    while i < len(lo) and j < len(sh):
        if lo[i] == sh[j]:
            i += 1
            j += 1
        elif not skipped:
            i += 1
            skipped = True
        else:
            return False
    return True
```

Then modify the `PredictRequest` class to add a `model_validator(mode="before")`:

Replace:

```python
class PredictRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    image_primary: str
    image_wrist: str | None = None
    proprio: list[float]
    instruction: str
    model_version: str | None = None
    t_mono_ns: dict[str, Any] | None = Field(default=None, alias="_t_mono_ns")

    @field_validator("instruction")
    @classmethod
    def _instruction_byte_length(cls, v: str) -> str:
        n = len(v.encode("utf-8"))
        if n > INSTRUCTION_MAX_BYTES:
            raise ValueError(
                f"instruction byte length {n} > {INSTRUCTION_MAX_BYTES} "
                f"(UTF-8 byte count, not char count)"
            )
        return v
```

With:

```python
class PredictRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    image_primary: str
    image_wrist: str | None = None
    proprio: list[float]
    instruction: str
    model_version: str | None = None
    t_mono_ns: dict[str, Any] | None = Field(default=None, alias="_t_mono_ns")

    @model_validator(mode="before")
    @classmethod
    def _typo_guard(cls, data: Any) -> Any:
        """F4: catch wire-level typos before extra='ignore' silently drops them.

        Two ordered rules; first match wins:
          1. Near-miss: any unknown key within Damerau-Levenshtein 1 of a
             modeled field → ValueError('did you mean X?').
          2. Image-prefix: any unknown key starting with 'image_' that didn't
             trip rule 1 → ValueError('unknown image field; known: ...').
        Other unknowns pass through and are dropped by extra='ignore'.
        """
        if not isinstance(data, dict):
            return data  # let pydantic handle non-dict inputs naturally
        for key in list(data.keys()):
            if key in _MODELED_FIELDS:
                continue
            # Rule 1: near-miss
            for modeled in _MODELED_FIELDS:
                if _damerau_levenshtein_within_one(key, modeled):
                    raise ValueError(
                        f"unknown field {key!r}; did you mean {modeled!r}?"
                    )
            # Rule 2: image_* prefix fallback
            if key.startswith("image_"):
                raise ValueError(
                    f"unknown image field {key!r}; "
                    f"known: image_primary, image_wrist"
                )
        return data

    @field_validator("instruction")
    @classmethod
    def _instruction_byte_length(cls, v: str) -> str:
        n = len(v.encode("utf-8"))
        if n > INSTRUCTION_MAX_BYTES:
            raise ValueError(
                f"instruction byte length {n} > {INSTRUCTION_MAX_BYTES} "
                f"(UTF-8 byte count, not char count)"
            )
        return v
```

Also update the import line at the top:

```python
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
```

- [ ] **Step 2.4: Run test to verify it passes**

Run: `PYTHONPATH= uv run pytest tests/test_validation_typo.py -v`

Expected: 9 PASS.

- [ ] **Step 2.5: Run existing schema tests to verify no regression**

Run: `PYTHONPATH= uv run pytest tests/test_deployment_schemas.py tests/test_validation_prompt.py -v`

Expected: All previous tests still pass. Most importantly, `test_predict_request_full_with_aliased_underscore_field` (in `test_deployment_schemas.py`) — which sends `_t_mono_ns` as a wire field — must still pass; `_t_mono_ns` is in `_MODELED_FIELDS`, so the typo guard does nothing for it.

- [ ] **Step 2.6: Commit**

```bash
git add src/vla_project/deployment/schemas.py tests/test_validation_typo.py
git commit -m "$(cat <<'EOF'
feat(deployment): F4 wire-typo guard via Damerau-Levenshtein

Catch typos like image_pirmary / propio / model_versionn at the schema
boundary, while preserving extra='ignore' for future MimicRec observability
fields (_request_id, _trace, _session_token).

Two ordered rules in model_validator(mode='before'):
  1. Damerau-Levenshtein distance ≤ 1 from a modeled field → 'did you mean X?'
  2. image_* prefix without near miss → 'unknown image field; known: ...'

Damerau (not plain Levenshtein) catches single-char transpositions, which
are the typical typo class: pirmary, wirst, mnoo all have plain-Levenshtein
distance 2 but Damerau distance 1.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: F1 — Image resolution sanity bound

**Files:**
- Modify: `src/vla_project/deployment/domain_adapter.py`
- Create: `tests/test_validation_image_sanity.py`

- [ ] **Step 3.1: Write the failing test**

Create `tests/test_validation_image_sanity.py`:

```python
"""F1: image resolution sanity bound, header-parse-first to avoid pixel-decode
allocation on absurdly-large images.

Spec: docs/superpowers/specs/2026-05-08-server-request-validation-design.md §F1.
Bound: 64 ≤ side ≤ 4096. Outside → ValueError (→ 422 at HTTP layer).

Heavy boundary cases (4096×4096) live as unit tests on _decode_jpeg_b64
directly to avoid full FastAPI round-trips for large images.
"""
import base64
import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from vla_project.deployment.domain_adapter import (
    DomainAdapter,
    IMAGE_MAX_SIDE,
    IMAGE_MIN_SIDE,
)
from vla_project.deployment.inference_server import build_app

DEPLOY_YAML = "configs/deploy/v36_libero_spatial.yaml"


def _b64_jpeg_size(w: int, h: int) -> str:
    img = Image.new("RGB", (w, h), color=(127, 127, 127))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=70)
    return base64.b64encode(buf.getvalue()).decode("ascii")


# ----- Unit-level: _decode_jpeg_b64 direct call (heavy boundaries here) -----

def test_decode_32x32_rejected():
    with pytest.raises(ValueError) as exc:
        DomainAdapter._decode_jpeg_b64(_b64_jpeg_size(32, 32))
    assert "out of sanity bound" in str(exc.value).lower()


def test_decode_5000x5000_rejected():
    with pytest.raises(ValueError) as exc:
        DomainAdapter._decode_jpeg_b64(_b64_jpeg_size(5000, 5000))
    assert "out of sanity bound" in str(exc.value).lower()


def test_decode_64x64_boundary_accepted():
    img = DomainAdapter._decode_jpeg_b64(_b64_jpeg_size(IMAGE_MIN_SIDE, IMAGE_MIN_SIDE))
    assert img.shape == (IMAGE_MIN_SIDE, IMAGE_MIN_SIDE, 3)


def test_decode_4096x4096_boundary_accepted():
    img = DomainAdapter._decode_jpeg_b64(_b64_jpeg_size(IMAGE_MAX_SIDE, IMAGE_MAX_SIDE))
    assert img.shape == (IMAGE_MAX_SIDE, IMAGE_MAX_SIDE, 3)


def test_decode_4097x4097_rejected():
    with pytest.raises(ValueError) as exc:
        DomainAdapter._decode_jpeg_b64(_b64_jpeg_size(IMAGE_MAX_SIDE + 1, IMAGE_MAX_SIDE + 1))
    assert "out of sanity bound" in str(exc.value).lower()


def test_decode_anisotropic_one_dim_too_small_rejected():
    """480×32 — width OK, height < min."""
    with pytest.raises(ValueError):
        DomainAdapter._decode_jpeg_b64(_b64_jpeg_size(480, 32))


# ----- Integration-level: full FastAPI request path -----

@pytest.fixture
def client():
    app = build_app(
        predictor_kind="hold_position",
        checkpoint=None,
        deploy_config_path=DEPLOY_YAML,
        domain_id=0,
    )
    return TestClient(app)


def test_request_with_32x32_image_returns_422(client):
    body = {
        "image_primary": _b64_jpeg_size(32, 32),
        "proprio": [0.0] * 7,  # v36_libero_spatial.yaml proprio.source.total_dim
        "instruction": "x",
    }
    resp = client.post("/predict", json=body)
    assert resp.status_code == 422
    assert "out of sanity bound" in str(resp.json()["detail"]).lower()


def test_request_with_224x224_image_returns_200(client):
    body = {
        "image_primary": _b64_jpeg_size(224, 224),
        "proprio": [0.0] * 7,  # v36_libero_spatial.yaml proprio.source.total_dim
        "instruction": "x",
    }
    resp = client.post("/predict", json=body)
    assert resp.status_code == 200
```

- [ ] **Step 3.2: Run test to verify it fails**

Run: `PYTHONPATH= uv run pytest tests/test_validation_image_sanity.py -v`

Expected: FAIL — `from vla_project.deployment.domain_adapter import IMAGE_MAX_SIDE, IMAGE_MIN_SIDE` fails (constants don't exist yet). This is intentional; once the constants are added, the rejection tests will fail because the bound isn't enforced.

- [ ] **Step 3.3: Add constants + modify `_decode_jpeg_b64` in domain_adapter.py**

In `src/vla_project/deployment/domain_adapter.py`, add module-level constants near the top of the file (after the imports, before the `class HardFailAssertion` line):

```python
# F1 (per docs/superpowers/specs/2026-05-08-server-request-validation-design.md §F1):
# Image side sanity bounds. Catches replay corruption (1×1) and abusive payloads
# (100k×100k) before the JPEG decoder is asked to allocate pixel buffers.
IMAGE_MIN_SIDE: int = 64
IMAGE_MAX_SIDE: int = 4096
```

Then replace `_decode_jpeg_b64`:

Replace:

```python
    @staticmethod
    def _decode_jpeg_b64(b64_str: str) -> np.ndarray:
        raw = base64.b64decode(b64_str)
        img = Image.open(io.BytesIO(raw)).convert("RGB")
        return np.asarray(img, dtype=np.uint8)
```

With:

```python
    @staticmethod
    def _decode_jpeg_b64(b64_str: str) -> np.ndarray:
        raw = base64.b64decode(b64_str)
        # F1: header-parse-first. Image.open() reads only the JPEG header
        # (no pixel decode); .size returns (W, H) from the header. We bound
        # the dimensions before convert("RGB") forces full pixel decode,
        # so an attacker / corrupt payload can't allocate gigabytes via
        # an oversized header.
        img = Image.open(io.BytesIO(raw))
        w, h = img.size
        if min(w, h) < IMAGE_MIN_SIDE or max(w, h) > IMAGE_MAX_SIDE:
            raise ValueError(
                f"image side ({w}, {h}) out of sanity bound "
                f"[{IMAGE_MIN_SIDE}, {IMAGE_MAX_SIDE}]"
            )
        img = img.convert("RGB")
        return np.asarray(img, dtype=np.uint8)
```

- [ ] **Step 3.4: Run test to verify it passes**

Run: `PYTHONPATH= uv run pytest tests/test_validation_image_sanity.py -v`

Expected: 8 PASS (6 unit + 2 integration).

- [ ] **Step 3.5: Run existing tests to verify no regression**

Run: `PYTHONPATH= uv run pytest tests/test_domain_adapter.py tests/test_serve_smoke.py tests/test_inference_server_minimal.py -v`

Expected: All pass (existing tests use 224×224 which is within bounds).

- [ ] **Step 3.6: Commit**

```bash
git add src/vla_project/deployment/domain_adapter.py tests/test_validation_image_sanity.py
git commit -m "$(cat <<'EOF'
feat(deployment): F1 image-side sanity bound (header-parse-first)

Add IMAGE_MIN_SIDE=64 / IMAGE_MAX_SIDE=4096 module constants and check
PIL Image.size before convert("RGB"). Header-parse-first ordering bounds
worst-case memory: an oversized JPEG header is rejected before the decoder
allocates pixel buffers.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: F3 — Proprio NaN/inf reject + OOD warn/hard-reject

**Files:**
- Modify: `src/vla_project/deployment/domain_adapter.py`
- Create: `tests/test_validation_proprio.py`

- [ ] **Step 4.1: Write the failing test**

Create `tests/test_validation_proprio.py`:

```python
"""F3: proprio non-finite reject + OOD warn / hard-reject.

Spec: docs/superpowers/specs/2026-05-08-server-request-validation-design.md §F3.
Three behaviors:
  - F3a: NaN/inf → 422 unconditionally.
  - F3b warn: |normed| > PROPRIO_OOD_WARN_ABS (1.0) → control flow continues
    (clip absorbs), structured WARNING log with event=proprio_ood emitted.
  - F3b hard: |normed| > PROPRIO_OOD_HARD_ABS (10.0) → 422 with msg containing
    'unit mismatch'. Hard reject runs first; no proprio_ood warn is emitted
    when raising.
"""
import base64
import io
import json
import logging
import math

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from vla_project.deployment.inference_server import build_app

DEPLOY_YAML = "configs/deploy/v36_libero_spatial.yaml"


def _b64_jpeg() -> str:
    img = Image.new("RGB", (224, 224), color=(127, 127, 127))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=70)
    return base64.b64encode(buf.getvalue()).decode("ascii")


@pytest.fixture
def client():
    """hold_position mode — does not normalize proprio (no norm_stats), so OOD
    tests need xvla_adapter mode. We use hold_position only for the NaN/inf path
    which fires before normalization."""
    app = build_app(
        predictor_kind="hold_position",
        checkpoint=None,
        deploy_config_path=DEPLOY_YAML,
        domain_id=0,
    )
    return TestClient(app)


# ----- F3a: non-finite proprio (any predictor mode) -----

def test_proprio_nan_returns_422(client):
    """v36_libero_spatial.yaml has proprio.source.total_dim == 7."""
    proprio = [0.0] * 7
    proprio[3] = float("nan")
    body = {"image_primary": _b64_jpeg(), "proprio": proprio, "instruction": "x"}
    resp = client.post("/predict", json=body)
    # FastAPI / pydantic may accept NaN in list[float] via JSON parser; the
    # check happens server-side in DomainAdapter.preprocess. Either way the
    # final HTTP code must be 422.
    assert resp.status_code == 422
    detail = str(resp.json()["detail"]).lower()
    assert "non-finite" in detail or "nan" in detail or "infinite" in detail


def test_proprio_inf_returns_422(client):
    proprio = [0.0] * 7
    proprio[6] = float("inf")
    body = {"image_primary": _b64_jpeg(), "proprio": proprio, "instruction": "x"}
    resp = client.post("/predict", json=body)
    assert resp.status_code == 422


# ----- F3b: OOD warn + hard reject (xvla_adapter mode required for norm_stats) -----
#
# We exercise _normalize_proprio directly with synthetic norm_stats rather than
# spinning up xvla_adapter mode (which requires a ckpt export dir). This keeps
# the test independent of test fixtures for the broader xvla_adapter setup.

import numpy as np

from vla_project.deployment.domain_adapter import (
    DomainAdapter,
    PROPRIO_OOD_HARD_ABS,
    PROPRIO_OOD_WARN_ABS,
    DeployConfig,
    load_deploy_config,
)


@pytest.fixture
def adapter_with_norm():
    """Build a DomainAdapter with synthetic q01/q99 spanning [-1, +1] so that
    raw proprio values map directly to `normed` of the same magnitude.

    With q01 = [-1]*8, q99 = [+1]*8, span = 2, the normalize formula
    `2 * (x - q01) / span - 1 = x` so raw == normed. This makes the test
    threshold values readable: feeding x=1.5 produces normed=1.5.
    """
    cfg = load_deploy_config(DEPLOY_YAML)
    norm_stats = {
        "proprio": {
            "q01": [-1.0] * cfg.proprio.adapt.output_dim,
            "q99": [+1.0] * cfg.proprio.adapt.output_dim,
            "mean": [0.0] * cfg.proprio.adapt.output_dim,
            "mask": [True] * cfg.proprio.adapt.output_dim,
        },
        "action": {
            "q01": [-1.0] * cfg.ckpt.expected_action_dim,
            "q99": [+1.0] * cfg.ckpt.expected_action_dim,
            "mean": [0.0] * cfg.ckpt.expected_action_dim,
            "mask": [True] * cfg.ckpt.expected_action_dim,
        },
    }
    return DomainAdapter(cfg=cfg, norm_stats=norm_stats, domain_id=0)


def test_normalize_in_range_no_warn(adapter_with_norm, caplog):
    x = np.zeros(adapter_with_norm.cfg.proprio.adapt.output_dim, dtype=np.float32)
    with caplog.at_level(logging.WARNING):
        out = adapter_with_norm._normalize_proprio(x)
    assert out.shape == x.shape
    assert not any("proprio_ood" in r.message for r in caplog.records)


def test_normalize_excess_1p5_emits_warn_and_clips(adapter_with_norm, caplog):
    """|normed|=1.5: above WARN threshold (1.0), below HARD threshold (10.0).
    Expected: WARNING log with event=proprio_ood, output clipped to ±1."""
    x = np.zeros(adapter_with_norm.cfg.proprio.adapt.output_dim, dtype=np.float32)
    x[2] = 1.5  # any single dim out
    with caplog.at_level(logging.WARNING):
        out = adapter_with_norm._normalize_proprio(x)
    # Clipped to ±1
    assert abs(out[2]) <= 1.0
    # Warn fired
    warns = [r for r in caplog.records if "proprio_ood" in r.message]
    assert len(warns) == 1
    payload = json.loads(warns[0].message)
    assert payload["event"] == "proprio_ood"
    assert 2 in payload["ood_dims"]
    assert payload["ood_dim_count"] >= 1


def test_normalize_excess_10_boundary_warn_only(adapter_with_norm, caplog):
    """|normed|=10.0 exactly: at HARD boundary; spec says > 10 is hard, so
    exactly 10 is warn-only (not raised)."""
    x = np.zeros(adapter_with_norm.cfg.proprio.adapt.output_dim, dtype=np.float32)
    x[0] = PROPRIO_OOD_HARD_ABS  # 10.0
    with caplog.at_level(logging.WARNING):
        out = adapter_with_norm._normalize_proprio(x)
    assert abs(out[0]) <= 1.0
    assert any("proprio_ood" in r.message for r in caplog.records)


def test_normalize_excess_above_hard_raises(adapter_with_norm, caplog):
    """|normed|=11.0: above HARD threshold (10.0). Expected: ValueError with
    'unit mismatch' message; NO proprio_ood warn (per F3 ordering)."""
    x = np.zeros(adapter_with_norm.cfg.proprio.adapt.output_dim, dtype=np.float32)
    x[5] = 11.0
    with caplog.at_level(logging.WARNING):
        with pytest.raises(ValueError) as exc:
            adapter_with_norm._normalize_proprio(x)
    assert "unit mismatch" in str(exc.value).lower()
    assert "5" in str(exc.value)  # dim 5 named
    # F3 ordering: hard reject suppresses the proprio_ood warn line
    assert not any("proprio_ood" in r.message for r in caplog.records)


def test_normalize_excess_just_above_hard_raises(adapter_with_norm):
    x = np.zeros(adapter_with_norm.cfg.proprio.adapt.output_dim, dtype=np.float32)
    x[0] = PROPRIO_OOD_HARD_ABS + 0.01  # 10.01
    with pytest.raises(ValueError):
        adapter_with_norm._normalize_proprio(x)
```

- [ ] **Step 4.2: Run test to verify it fails**

Run: `PYTHONPATH= uv run pytest tests/test_validation_proprio.py -v`

Expected: imports fail (`PROPRIO_OOD_HARD_ABS`, `PROPRIO_OOD_WARN_ABS` don't exist). After step 4.3 adds them, the rejection / warn-fire tests will fail.

- [ ] **Step 4.3: Add constants + modify `preprocess` and `_normalize_proprio`**

In `src/vla_project/deployment/domain_adapter.py`:

Add to the imports block at the top:

```python
import json
import logging
```

Add module-level constants (next to the F1 constants you added in Task 3):

```python
# F3 (per docs/superpowers/specs/2026-05-08-server-request-validation-design.md §F3):
# Proprio out-of-distribution thresholds. Computed against the normalized values
# (after q01/q99 mapping). >WARN absorbed by clip + WARNING log; >HARD raises.
# 10.0 is wide enough to admit legitimate startup poses outside training
# support but still catches deg/rad swap (rad ≈ 0.5 → deg = 30 → ~30x q-range).
PROPRIO_OOD_WARN_ABS: float = 1.0
PROPRIO_OOD_HARD_ABS: float = 10.0
```

Add a module-level logger near the top (after imports):

```python
logger = logging.getLogger("vla_project.deployment.domain_adapter")
```

Modify `preprocess` to add the NaN/inf check. Replace:

```python
        proprio_raw = np.asarray(req.proprio, dtype=np.float32)
        if proprio_raw.shape[0] != self.cfg.proprio.source.total_dim:
            raise ValueError(
                f"proprio length {proprio_raw.shape[0]} != "
                f"deploy.proprio.source.total_dim {self.cfg.proprio.source.total_dim}"
            )
```

With:

```python
        proprio_raw = np.asarray(req.proprio, dtype=np.float32)
        # F3a: non-finite proprio is unconditionally invalid. Catches NaN/inf
        # from upstream sensor faults or test fixtures; also short-circuits any
        # downstream normalize/clip that would silently swallow the signal.
        if not np.isfinite(proprio_raw).all():
            bad_dims = np.where(~np.isfinite(proprio_raw))[0].tolist()
            raise ValueError(
                f"proprio contains non-finite values at dims {bad_dims}"
            )
        if proprio_raw.shape[0] != self.cfg.proprio.source.total_dim:
            raise ValueError(
                f"proprio length {proprio_raw.shape[0]} != "
                f"deploy.proprio.source.total_dim {self.cfg.proprio.source.total_dim}"
            )
```

Modify `_normalize_proprio`. Replace:

```python
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
```

With:

```python
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
        # F3b: OOD detection happens BEFORE clip. Mask=False dims are not
        # subject to OOD checks (the model receives the raw value for them).
        abs_normed = np.abs(normed)
        # F3b hard: |normed| > PROPRIO_OOD_HARD_ABS → 422. Runs first so the
        # warn line below is skipped on hard reject (single invalid_request
        # log is sufficient — see spec §F3 "warn-vs-raise ordering").
        hard_violations = (abs_normed > PROPRIO_OOD_HARD_ABS) & mask
        if hard_violations.any():
            hard_dims = np.where(hard_violations)[0].tolist()
            max_excess = float((abs_normed * mask).max() - 1.0)
            raise ValueError(
                f"proprio normalized |x|>{PROPRIO_OOD_HARD_ABS} at dims "
                f"{hard_dims} (max excess {max_excess:.2f}); likely unit "
                f"mismatch (deg/rad swap or wrong proprio_key)"
            )
        # F3b warn: |normed| > PROPRIO_OOD_WARN_ABS (and ≤ HARD) → log + clip.
        warn_violations = (abs_normed > PROPRIO_OOD_WARN_ABS) & mask
        if warn_violations.any():
            ood_dims = np.where(warn_violations)[0].tolist()
            max_excess = float((abs_normed * mask).max() - 1.0)
            logger.warning(json.dumps({
                "event": "proprio_ood",
                "ood_dim_count": len(ood_dims),
                "ood_max_excess": round(max_excess, 3),
                "ood_dims": ood_dims,
            }))
        # Clamp to [-1, +1] (training-time q99 clipping convention).
        normed = np.clip(normed, -1.0, 1.0)
        return np.where(mask, normed, x).astype(np.float32)
```

- [ ] **Step 4.4: Run test to verify it passes**

Run: `PYTHONPATH= uv run pytest tests/test_validation_proprio.py -v`

Expected: 7 PASS.

- [ ] **Step 4.5: Run existing tests to verify no regression**

Run: `PYTHONPATH= uv run pytest tests/test_domain_adapter.py tests/test_serve_smoke.py tests/test_inference_server_minimal.py tests/test_predictor_holdposition.py -v`

Expected: All pass (existing tests use proprio values in q-range, no NaN/inf).

- [ ] **Step 4.6: Commit**

```bash
git add src/vla_project/deployment/domain_adapter.py tests/test_validation_proprio.py
git commit -m "$(cat <<'EOF'
feat(deployment): F3 proprio non-finite reject + OOD warn / hard reject

Add three checks in DomainAdapter:
  - F3a: NaN / inf in raw proprio → 422 unconditionally.
  - F3b warn: |normed| > 1.0 → WARNING log with event=proprio_ood (mask-aware).
  - F3b hard: |normed| > 10.0 → 422 with 'unit mismatch' message.

Hard reject runs first (skips the warn line — the existing 422 invalid_request
handler logs the event once). Clip behavior unchanged: same training-time
Q99 convention, applied after the OOD detection step.

Constants PROPRIO_OOD_WARN_ABS / PROPRIO_OOD_HARD_ABS at module level per spec.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: F5 — `/admin/schema` introspection endpoint

**Files:**
- Modify: `src/vla_project/deployment/inference_server.py`
- Create: `tests/test_admin_schema.py`

- [ ] **Step 5.1: Write the failing test**

Create `tests/test_admin_schema.py`:

```python
"""F5: GET /admin/schema returns the wire contract introspection payload.

Spec: docs/superpowers/specs/2026-05-08-server-request-validation-design.md §F5.
Returns 9 top-level keys (predictor, ckpt, wrist_hard_required, request_fields,
proprio, image, instruction, prompt, proprio_ood). prompt.max_tokens is null in
both Phase 0 modes (Phase 0 deferral — server doesn't tokenize yet).
"""
import pytest
from fastapi.testclient import TestClient

from vla_project.deployment.domain_adapter import (
    IMAGE_MAX_SIDE,
    IMAGE_MIN_SIDE,
    PROPRIO_OOD_HARD_ABS,
    PROPRIO_OOD_WARN_ABS,
)
from vla_project.deployment.inference_server import build_app
from vla_project.deployment.schemas import INSTRUCTION_MAX_BYTES

DEPLOY_YAML = "configs/deploy/v36_libero_spatial.yaml"


@pytest.fixture
def hold_position_client():
    app = build_app(
        predictor_kind="hold_position",
        checkpoint=None,
        deploy_config_path=DEPLOY_YAML,
        domain_id=0,
    )
    return TestClient(app)


def test_admin_schema_route_exists(hold_position_client):
    resp = hold_position_client.get("/admin/schema")
    assert resp.status_code == 200


def test_admin_schema_response_has_expected_top_level_keys(hold_position_client):
    resp = hold_position_client.get("/admin/schema")
    data = resp.json()
    expected_keys = {
        "predictor",
        "ckpt",
        "wrist_hard_required",
        "request_fields",
        "proprio",
        "image",
        "instruction",
        "prompt",
        "proprio_ood",
    }
    assert set(data.keys()) == expected_keys


def test_admin_schema_predictor_is_hold_position_class_name(hold_position_client):
    data = hold_position_client.get("/admin/schema").json()
    assert data["predictor"] == "HoldPositionChunkPredictor"


def test_admin_schema_ckpt_subfields(hold_position_client):
    data = hold_position_client.get("/admin/schema").json()
    ckpt = data["ckpt"]
    assert "expected_unnorm_key" in ckpt
    assert "expected_action_chunk_len" in ckpt
    assert "expected_action_dim" in ckpt
    assert "expected_proprio_dim" in ckpt
    assert ckpt["expected_action_dim"] == 7  # MimicRec MVP


def test_admin_schema_image_bounds_match_constants(hold_position_client):
    data = hold_position_client.get("/admin/schema").json()
    assert data["image"]["min_side"] == IMAGE_MIN_SIDE
    assert data["image"]["max_side"] == IMAGE_MAX_SIDE


def test_admin_schema_instruction_max_bytes_matches_constant(hold_position_client):
    data = hold_position_client.get("/admin/schema").json()
    assert data["instruction"]["max_bytes"] == INSTRUCTION_MAX_BYTES


def test_admin_schema_proprio_ood_thresholds_match_constants(hold_position_client):
    data = hold_position_client.get("/admin/schema").json()
    assert data["proprio_ood"]["warn_threshold"] == PROPRIO_OOD_WARN_ABS
    assert data["proprio_ood"]["hard_threshold"] == PROPRIO_OOD_HARD_ABS


def test_admin_schema_prompt_max_tokens_null_in_phase_0(hold_position_client):
    """Phase 0 deferral: server does not tokenize, so reporting max_tokens
    would be misleading. Should be null in both predictor modes."""
    data = hold_position_client.get("/admin/schema").json()
    assert data["prompt"]["max_tokens"] is None


def test_admin_schema_request_fields_match_deploy_yaml(hold_position_client):
    """Names come from configs/deploy/v36_libero_spatial.yaml's request_fields."""
    data = hold_position_client.get("/admin/schema").json()
    rf = data["request_fields"]
    assert rf["scene_image"] == "image_primary"
    assert rf["proprio"] == "proprio"
    assert rf["instruction"] == "instruction"


def test_admin_schema_proprio_source_components_present(hold_position_client):
    data = hold_position_client.get("/admin/schema").json()
    src = data["proprio"]["source"]
    assert "components" in src
    assert "total_dim" in src
    assert isinstance(src["components"], list)
    assert all("name" in c and "dims" in c and "units" in c for c in src["components"])
    assert sum(c["dims"] for c in src["components"]) == src["total_dim"]


def test_admin_schema_wrist_hard_required_is_bool(hold_position_client):
    data = hold_position_client.get("/admin/schema").json()
    assert isinstance(data["wrist_hard_required"], bool)
    # hold_position mode never hard-requires wrist (no ckpt cfg → False)
    assert data["wrist_hard_required"] is False


# ----- xvla_adapter mode (synthetic ckpt) -----

import json
from pathlib import Path


def _write_synthetic_ckpt(tmp_path: Path, *, use_wrist_bridge: bool = False) -> Path:
    """Phase 0 ckpt with the minimal meta.json keys the validator demands.
    Pattern copied from test_serve_smoke.py:test_predict_hard_required_wrist_missing."""
    ckpt_dir = tmp_path / "fake_v36"
    ckpt_dir.mkdir()
    meta = {
        "step": 0,
        "cfg": {
            "model": {
                "num_domains": 1,
                "use_wrist_bridge": use_wrist_bridge,
                "wrist_in_llm": False,
                "wrist_view_dropout_p": 0.0,
            },
            "data": {"unnorm_key": "libero_spatial_no_noops", "action_chunk_len": 8},
        },
        "norm_stats": {
            "libero_spatial_no_noops": {
                "action": {"mean": [0.0]*7, "std": [1.0]*7, "q01": [-1.0]*7,
                           "q99": [1.0]*7, "mask": [True]*6 + [False]},
                "proprio": {"mean": [0.0]*8, "std": [1.0]*8, "q01": [-1.0]*8,
                            "q99": [1.0]*8, "mask": [True]*8},
            }
        },
    }
    (ckpt_dir / "meta.json").write_text(json.dumps(meta))
    return ckpt_dir


@pytest.fixture
def xvla_adapter_client(tmp_path):
    ckpt_dir = _write_synthetic_ckpt(tmp_path, use_wrist_bridge=False)
    app = build_app(
        predictor_kind="xvla_adapter",
        checkpoint=ckpt_dir,
        deploy_config_path=DEPLOY_YAML,
        domain_id=0,
    )
    return TestClient(app)


@pytest.fixture
def xvla_adapter_wrist_required_client(tmp_path):
    ckpt_dir = _write_synthetic_ckpt(tmp_path, use_wrist_bridge=True)
    app = build_app(
        predictor_kind="xvla_adapter",
        checkpoint=ckpt_dir,
        deploy_config_path=DEPLOY_YAML,
        domain_id=0,
    )
    return TestClient(app)


def test_admin_schema_predictor_is_xvla_adapter_class_name(xvla_adapter_client):
    data = xvla_adapter_client.get("/admin/schema").json()
    assert data["predictor"] == "XVLAAdapterChunkPredictor"


def test_admin_schema_prompt_max_tokens_null_in_xvla_adapter_phase_0(xvla_adapter_client):
    """Phase 0 deferral: same null contract regardless of predictor mode."""
    data = xvla_adapter_client.get("/admin/schema").json()
    assert data["prompt"]["max_tokens"] is None


def test_admin_schema_wrist_hard_required_true_when_ckpt_uses_wrist_bridge(
    xvla_adapter_wrist_required_client,
):
    data = xvla_adapter_wrist_required_client.get("/admin/schema").json()
    assert data["wrist_hard_required"] is True
```

- [ ] **Step 5.2: Run test to verify it fails**

Run: `PYTHONPATH= uv run pytest tests/test_admin_schema.py -v`

Expected: All 14 tests FAIL with 404 from `/admin/schema` (route does not exist).

- [ ] **Step 5.3: Add the route to `inference_server.py`**

In `src/vla_project/deployment/inference_server.py`:

Add to the imports block at the top (additions only):

```python
from vla_project.deployment.domain_adapter import (
    DomainAdapter,
    IMAGE_MAX_SIDE,
    IMAGE_MIN_SIDE,
    PROPRIO_OOD_HARD_ABS,
    PROPRIO_OOD_WARN_ABS,
    load_deploy_config,
)
from vla_project.deployment.schemas import (
    INSTRUCTION_MAX_BYTES,
    PredictRequest,
    PredictResponse,
)
```

(Replace the two existing import lines from `domain_adapter` and `schemas` with these.)

Add the route inside `build_app`, immediately after the `@app.get("/healthz")` block:

```python
    @app.get("/admin/schema")
    async def admin_schema() -> dict:
        """F5: wire-contract introspection. Returns the minimum a client needs
        to construct valid requests + sanity-check shapes against this server's
        loaded config. See spec §F5 for field semantics.

        prompt.max_tokens is null in BOTH predictor modes for Phase 0 — the
        server does not tokenize yet, so reporting a max_tokens value would
        imply an enforced ceiling that does not exist. Phase 1 will populate
        the field when XVLAAdapterChunkPredictor.predict() actually
        tokenizes the instruction.
        """
        return {
            "predictor": state["predictor_class"],
            "ckpt": {
                "expected_unnorm_key": cfg.ckpt.expected_unnorm_key,
                "expected_action_chunk_len": cfg.ckpt.expected_action_chunk_len,
                "expected_action_dim": cfg.ckpt.expected_action_dim,
                "expected_proprio_dim": cfg.ckpt.expected_proprio_dim,
            },
            "wrist_hard_required": wrist_hard_required,
            "request_fields": {
                "scene_image": cfg.request_fields.scene_image,
                "wrist_image": cfg.request_fields.wrist_image,
                "proprio": cfg.request_fields.proprio,
                "instruction": cfg.request_fields.instruction,
            },
            "proprio": {
                "source": {
                    "components": [
                        {"name": c.name, "dims": c.dims, "units": c.units}
                        for c in cfg.proprio.source.components
                    ],
                    "total_dim": cfg.proprio.source.total_dim,
                },
            },
            "image": {"min_side": IMAGE_MIN_SIDE, "max_side": IMAGE_MAX_SIDE},
            "instruction": {"max_bytes": INSTRUCTION_MAX_BYTES},
            "prompt": {"max_tokens": None},  # Phase 0 deferral — see docstring
            "proprio_ood": {
                "warn_threshold": PROPRIO_OOD_WARN_ABS,
                "hard_threshold": PROPRIO_OOD_HARD_ABS,
            },
        }
```

- [ ] **Step 5.4: Run test to verify it passes**

Run: `PYTHONPATH= uv run pytest tests/test_admin_schema.py -v`

Expected: 14 PASS (11 hold_position + 3 xvla_adapter mode).

- [ ] **Step 5.5: Run existing tests to verify no regression**

Run: `PYTHONPATH= uv run pytest tests/test_inference_server_minimal.py tests/test_serve_smoke.py -v`

Expected: All pass.

- [ ] **Step 5.6: Commit**

```bash
git add src/vla_project/deployment/inference_server.py tests/test_admin_schema.py
git commit -m "$(cat <<'EOF'
feat(deployment): F5 GET /admin/schema introspection endpoint

Return wire-contract facts a client needs to construct valid requests:
  - predictor class name
  - ckpt expected_* dims (action_dim, action_chunk_len, proprio_dim, unnorm_key)
  - wrist_hard_required flag
  - request_fields canonical wire names
  - proprio.source components + total_dim (from deploy yaml)
  - image / instruction / proprio_ood thresholds (server-side constants)
  - prompt.max_tokens = null in Phase 0 (deferred until tokenize-truncate
    path is implemented in XVLAAdapterChunkPredictor.predict, Phase 1)

No auth — server already binds 127.0.0.1 by default in Phase 0.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: Extended existing tests + acceptance gate update

**Files:**
- Modify: `tests/test_domain_adapter.py`
- Modify: `tests/test_serve_smoke.py`
- Modify: `tests/test_inference_server_minimal.py`
- Modify: `src/vla_project/deployment/README.md`

- [ ] **Step 6.1: Extend `tests/test_domain_adapter.py`**

The existing helper in this file is named `_make_jpeg_b64` (NOT `_b64_jpeg` — that's the helper in `test_serve_smoke.py`). Use the existing name. Append two test functions at the end of the file:

```python
def test_preprocess_rejects_nan_proprio():
    """F3a regression: NaN in raw proprio raises before normalization."""
    cfg = load_deploy_config("configs/deploy/v36_libero_spatial.yaml")
    adapter = DomainAdapter(cfg=cfg, norm_stats=None, domain_id=0)
    proprio = [0.0] * cfg.proprio.source.total_dim
    proprio[1] = float("nan")
    req = PredictRequest(
        image_primary=_make_jpeg_b64(),
        proprio=proprio,
        instruction="x",
    )
    with pytest.raises(ValueError, match="non-finite"):
        adapter.preprocess(req)


def test_decode_jpeg_b64_rejects_too_small_image():
    """F1 regression: 32×32 image rejected by header sanity bound."""
    import base64
    import io
    from PIL import Image
    img = Image.new("RGB", (32, 32))
    buf = io.BytesIO(); img.save(buf, format="JPEG", quality=70)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    with pytest.raises(ValueError, match="out of sanity bound"):
        DomainAdapter._decode_jpeg_b64(b64)
```

`PredictRequest` and `load_deploy_config` should already be imported at the top of the file (existing tests use them); no new imports needed.

- [ ] **Step 6.2: Extend `tests/test_serve_smoke.py`**

Append one test function at the end:

```python
def test_typo_image_pirmary_returns_422_end_to_end(client):
    """F4 regression via FastAPI: image_pirmary typo caught before extra='ignore'
    silently drops it. Exercises the full HTTP path including pydantic
    model_validator(mode='before')."""
    body = {
        "image_pirmary": _b64_jpeg(),  # typo of image_primary
        "image_primary": _b64_jpeg(),  # also send the correct one so we don't
                                        # also fail on missing required field
        "proprio": [0.0] * 8,
        "instruction": "test",
    }
    resp = client.post("/predict", json=body)
    assert resp.status_code == 422
    detail = resp.json()["detail"]
    # FastAPI's RequestValidationError serializes as a list of error dicts
    detail_str = str(detail).lower()
    assert "image_pirmary" in detail_str
    assert "did you mean" in detail_str or "image_primary" in detail_str
```

- [ ] **Step 6.3: Extend `tests/test_inference_server_minimal.py`**

This file has no shared `client` fixture — its existing `test_build_app_returns_fastapi_with_healthz` builds the app inline. Match that pattern. Append:

```python
def test_admin_schema_route_returns_200():
    """F5 regression: /admin/schema must respond alongside /healthz on a
    minimal hold_position-mode build_app."""
    app = build_app(
        predictor_kind="hold_position",
        checkpoint=None,
        deploy_config_path="configs/deploy/v36_libero_spatial.yaml",
        domain_id=0,
        inject_sleep_s=0.0,
    )
    client = TestClient(app)
    resp = client.get("/admin/schema")
    assert resp.status_code == 200
    assert "predictor" in resp.json()
```

- [ ] **Step 6.4: Update `src/vla_project/deployment/README.md` Phase 0 acceptance gate**

In `src/vla_project/deployment/README.md`, find the section "## Phase 0 acceptance verification" and replace its pytest command block:

```bash
PYTHONPATH= uv run pytest \
  tests/test_deployment_schemas.py \
  tests/test_domain_adapter.py \
  tests/test_predictor_base.py \
  tests/test_predictor_holdposition.py \
  tests/test_predictor_xvla_adapter.py \
  tests/test_runtime_load.py \
  tests/test_inference_server_minimal.py \
  tests/test_serve_smoke.py \
  -q
```

With:

```bash
PYTHONPATH= uv run pytest \
  tests/test_deployment_schemas.py \
  tests/test_domain_adapter.py \
  tests/test_predictor_base.py \
  tests/test_predictor_holdposition.py \
  tests/test_predictor_xvla_adapter.py \
  tests/test_runtime_load.py \
  tests/test_inference_server_minimal.py \
  tests/test_serve_smoke.py \
  tests/test_validation_image_sanity.py \
  tests/test_validation_prompt.py \
  tests/test_validation_proprio.py \
  tests/test_validation_typo.py \
  tests/test_admin_schema.py \
  -q
```

- [ ] **Step 6.5: Run the full extended acceptance gate**

Run:

```bash
PYTHONPATH= uv run pytest \
  tests/test_deployment_schemas.py \
  tests/test_domain_adapter.py \
  tests/test_predictor_base.py \
  tests/test_predictor_holdposition.py \
  tests/test_predictor_xvla_adapter.py \
  tests/test_runtime_load.py \
  tests/test_inference_server_minimal.py \
  tests/test_serve_smoke.py \
  tests/test_validation_image_sanity.py \
  tests/test_validation_prompt.py \
  tests/test_validation_proprio.py \
  tests/test_validation_typo.py \
  tests/test_admin_schema.py \
  -q
```

Expected: All tests pass (no exact count given — contributions are F1=8, F2=7, F3=7, F4=9, F5=14, plus existing 8 files ≈ 60+ tests; existing extended tests add ~4 more. Verify exit code 0).

- [ ] **Step 6.6: Commit**

```bash
git add tests/test_domain_adapter.py tests/test_serve_smoke.py tests/test_inference_server_minimal.py src/vla_project/deployment/README.md
git commit -m "$(cat <<'EOF'
test(deployment): regression cases + acceptance gate for F1-F5

- test_domain_adapter: add F3a NaN proprio + F1 image-bound regression cases.
- test_serve_smoke: add F4 image_pirmary typo end-to-end via FastAPI.
- test_inference_server_minimal: assert /admin/schema responds alongside /healthz.
- README: append the 5 new validation test files to the Phase 0 acceptance gate.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: Final verification — full acceptance gate + MimicRec smoke

**Files:** none modified.

- [ ] **Step 7.1: Run full Phase 0 acceptance gate**

Run the command from `src/vla_project/deployment/README.md` (the one updated in Task 6):

```bash
PYTHONPATH= uv run pytest \
  tests/test_deployment_schemas.py \
  tests/test_domain_adapter.py \
  tests/test_predictor_base.py \
  tests/test_predictor_holdposition.py \
  tests/test_predictor_xvla_adapter.py \
  tests/test_runtime_load.py \
  tests/test_inference_server_minimal.py \
  tests/test_serve_smoke.py \
  tests/test_validation_image_sanity.py \
  tests/test_validation_prompt.py \
  tests/test_validation_proprio.py \
  tests/test_validation_typo.py \
  tests/test_admin_schema.py \
  -q
```

Expected: exit code 0. No skipped tests in the new files.

- [ ] **Step 7.2: Run MimicRec integration smoke**

In a separate terminal (Terminal 1), start the server:

```bash
uv run python scripts/serve.py \
  --predictor hold_position \
  --deploy-config configs/deploy/v36_libero_spatial.yaml \
  --domain-id 0 --port 8001
```

In Terminal 2:

```bash
cp configs/deploy/mimicrec_pairing_example.yaml \
   /home/takakimaeda/MimicRec/configs/inference/x_vla_v36_smoke.yaml
cd /home/takakimaeda/MimicRec
.venv/bin/python scripts/smoke_inference_real_data.py
```

Expected output:

```
✅ inference mock pipeline works end-to-end with real data
IK failures: 0/N    # zero ee_delta → no IK displacement
```

This regression-checks that no validation rule (typo guard, byte length, image bound, NaN check) breaks the existing MimicRec wire format.

- [ ] **Step 7.3: Stop the server**

In Terminal 1: `Ctrl-C`.

- [ ] **Step 7.4: Run codex review against main (per project CLAUDE.md "Before opening or merging a PR")**

```bash
codex review --base main --title "Inference server F1-F5 request validation"
```

Per project CLAUDE.md, `--base main` is the form for pre-PR review of the full branch diff (covers all commits since main). If you have not yet committed all the work from Tasks 1-6, prefer `codex review --uncommitted` instead. If `codex` is unavailable in the environment, surface this to the user; do not silently skip.

Treat findings as a second opinion. Resolve via `superpowers:receiving-code-review` skill (verify each claim against code, technical correctness wins). Do not auto-apply suggestions without judgment.

- [ ] **Step 7.5: Mark spec acceptance checklist complete**

Open `docs/superpowers/specs/2026-05-08-server-request-validation-design.md` and mark the "Acceptance checklist" section's checkboxes that reflect the now-shipped state:

- [x] `schemas.py` keeps `extra="ignore"`; adds `instruction` byte validator + typo `model_validator`.
- [x] `domain_adapter.py` rejects NaN/inf proprio at preprocess entry.
- [x] `domain_adapter.py` checks image side bounds before pixel decode.
- [x] `domain_adapter.py` `_normalize_proprio` warn-logs at `>1` excess and raises at `>10`.
- [x] `inference_server.py` serves `GET /admin/schema` with `prompt.max_tokens=null` in both modes (Phase 0 deferral).
- [x] All 5 new test files green; 3 extended tests still green; existing 8 acceptance tests still green.
- [x] MimicRec smoke (`smoke_inference_real_data.py`) still green with no client-side change.
- [x] No new `configs/deploy/*.yaml` fields.
- [x] No new `predictors/*.py` constructor changes.
- [x] `tools/export_checkpoint.py` is NOT introduced as part of this change (tokenizer load remains Phase 1).

Then commit the spec update:

```bash
git add docs/superpowers/specs/2026-05-08-server-request-validation-design.md
git commit -m "$(cat <<'EOF'
docs(specs): mark request-validation acceptance checklist complete

All Phase 0 add-on validation features (F1 image sanity, F2 instruction
bytes, F3 proprio OOD, F4 wire-typo guard, F5 /admin/schema) shipped and
acceptance-gated. MimicRec smoke regression confirmed green.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Self-review notes

**Spec coverage check:**

| Spec section | Plan task |
|---|---|
| F1 Image resolution sanity bound | Task 3 |
| F2 Instruction byte-length sanity (Phase 0 part) | Task 1 |
| F2 Tokenize-time truncation (Phase 1 part) | (deferred; not in this plan) |
| F3 Proprio OOD warn + extreme-value hard reject | Task 4 |
| F4 Wire-typo guard | Task 2 |
| F5 `/admin/schema` introspection endpoint | Task 5 |
| Section 2 component changes (file-by-file) | Tasks 1-5 cover each file |
| Section 3 testing strategy | Tasks 1-6 (5 new files + 3 extensions + acceptance gate update) |
| Section 4 error handling table | Implicit in test assertions across Tasks 1-5 |
| Section 5 out of scope | Honored (no tokenizer load, no yaml changes, no prompt_processor.py) |
| Acceptance checklist | Task 7 step 7.5 |

**Type / signature consistency check:**

- `INSTRUCTION_MAX_BYTES` (Task 1) — used in `schemas.py` validator, also re-imported by `inference_server.py` in Task 5. ✓
- `IMAGE_MIN_SIDE` / `IMAGE_MAX_SIDE` (Task 3) — used in `_decode_jpeg_b64`, re-imported by `inference_server.py` in Task 5. ✓
- `PROPRIO_OOD_WARN_ABS` / `PROPRIO_OOD_HARD_ABS` (Task 4) — used in `_normalize_proprio`, re-imported by `inference_server.py` in Task 5 and by `tests/test_validation_proprio.py`. ✓
- `_MODELED_FIELDS` / `_damerau_levenshtein_within_one` (Task 2) — module-private, used only inside `schemas.py`. ✓
- `DomainAdapter._decode_jpeg_b64` signature unchanged (still `@staticmethod`, still takes `b64_str: str`, still returns `np.ndarray`). ✓
- `DomainAdapter.__init__` signature unchanged (no `tokenizer` / `prompt_max_len` per Task 1 file-structure note). ✓

**Placeholder scan:** no TBD / TODO / "implement later" / vague rules. Every step has either exact code or an exact command.
