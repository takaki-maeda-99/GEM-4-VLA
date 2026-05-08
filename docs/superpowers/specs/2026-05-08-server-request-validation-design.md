# VLA Inference Server — Request Validation (Phase 0 add-on) — Design

**Status:** ready for user review (codex pre-spec brainstorm review converged; spec-level codex review pending)
**Date:** 2026-05-08
**Owner:** Takaki Maeda
**Builds on:** `docs/superpowers/specs/2026-05-06-vla-inference-server-design.md` (Phase 0 server design)

## Context

The Phase 0 inference server (`src/vla_project/deployment/`) accepts `POST /predict` and runs a chain of validations + preprocessing inside `DomainAdapter.preprocess`. The shipped checks already enforce the high-confidence wire-format invariants (proprio length vs `proprio.source.total_dim`, `wrist_hard_required`, action row width on output). However, several classes of "obvious mistake" inputs reach the predictor unchallenged:

- Image whose decoded resolution is far outside what real cameras emit (e.g., 1×1, 100k×100k due to encoding bug or replay corruption).
- Instruction string whose byte length is degenerate (≥10 KB; empty is explicitly allowed per Phase 0 spec L153 "may be empty in pre-start states").
- Proprio whose values are at training-distribution edge (silently clipped to ±1) or completely off-scale (deg-vs-rad swap, `proprio_key` misconfiguration), with no log signal whatsoever.
- Wire-level typos (`image_wirst`, `image_pirmary`) silently dropped by `extra="ignore"` and never surfaced.
- No machine-readable way for clients to ask "what shape do you expect?" — they have to read the deploy yaml manually.

This add-on specifies five complementary validation features that close those gaps **without changing what the model sees** (clipping/normalization/resize logic is unchanged: those are Phase 1 / model-side concerns) and without adding new deploy yaml fields.

### Architectural principle (load-bearing for every decision below)

> **The VLA — i.e. the inference server, since it hosts the ckpt — owns all preprocessing (norm, resize, tokenize). The deploy yaml describes raw input shape; it does NOT carry preprocessing rules.**
>
> **Validation is "obvious mistake / sanity bound" only.** Reject for unit mismatches, typos, NaN/inf, and resource-abuse magnitudes. **Never** reject for being at the edge of training distribution; only warn-log it. The model's clipping/normalization absorbs distribution-edge inputs by design.

This principle is what disqualifies, e.g., `image.expected_size: [H, W]` in the deploy yaml (would push preprocessing knowledge yaml-side) and `proprio.normalized=true` from the client (would push norm_stats client-side).

### Decisions locked (post-codex brainstorm review)

| # | Decision |
|---|---|
| 1 | All 5 features ship in **Phase 0**, except (2)'s tokenize-time truncation, which is **deferred to Phase 1** (predictor doesn't actually use the tokens until then; pulling tokenizer load forward adds dependency surface for zero functional benefit). |
| 2 | No new fields in `configs/deploy/*.yaml`. All thresholds are either ckpt-derived or universal sanity constants. |
| 3 | `instruction = ""` is valid (matches Phase 0 spec L153 "may be empty in pre-start states"). Only the upper bound (≤10 000 UTF-8 bytes) is enforced. |
| 4 | `extra="ignore"` is preserved on `PredictRequest` — preserves forward-compat with future MimicRec observability fields (`_request_id`, `_trace`, etc.). Typos are caught by a targeted guard, not blanket rejection. |
| 5 | Proprio Q99 clipping behavior is unchanged (matches training-time clipping). The `>10x` hard reject is a unit-mismatch guard, not a distribution check. |
| 6 | `/admin/schema` returns the minimum needed for clients to construct valid requests + sanity-check shapes. Internal runtime fields (`device`, `dtype`, `step`) are excluded. |

---

## Section 1. Per-feature design

### F1. Image resolution sanity bound

**Problem:** `_decode_jpeg_b64` accepts any size PIL can decode. A 1×1 image will pass through to the SiglipImageTransform (Phase 1) and either crash there or produce garbage features.

**Behavior:** in `DomainAdapter._decode_jpeg_b64`:

1. `Image.open(io.BytesIO(raw))` — header parse only; do NOT call `.convert("RGB")` yet (would force pixel decode).
2. Read `.size` → `(W, H)`. If `min(W, H) < 64` or `max(W, H) > 4096` → `ValueError(f"image side {(W, H)} out of sanity bound [64, 4096]")`.
3. Then `convert("RGB")` and continue.

**Rationale:** real cameras emit 256×256 (LIBERO sim), 480×640 (RGBD heads), 720×1280 (USB cams). The bound `[64, 4096]` covers all plausible upstreams while catching `1×1` (replay bug) and `>4096` (encoding bug, abuse). The check before pixel decode also bounds memory: `Image.MAX_IMAGE_PIXELS` would otherwise let a 100k×100k header allocate 30 GB.

**No yaml field.** Each robot's actual camera resolution is *not* validated by the server — the SiglipImageTransform (Phase 1, server-side) handles resize/crop, per the architectural principle.

### F2. Instruction byte-length sanity (Phase 0) + tokenize-time truncation (Phase 1)

**Problem:** `instruction: str` accepts arbitrarily long strings. A 10 MB instruction will sit in memory and (in Phase 1) trigger Gemma tokenizer to allocate proportionally.

**Behavior (Phase 0):**

1. `PredictRequest.instruction: str` — `min_length` not set (empty allowed).
2. Custom `field_validator("instruction")`: `if len(value.encode("utf-8")) > INSTRUCTION_MAX_BYTES: raise ValueError(f"instruction byte length {n} > {INSTRUCTION_MAX_BYTES}")` where `INSTRUCTION_MAX_BYTES = 10_000`. Pydantic's `max_length` constrains characters, not bytes; UTF-8 multibyte (Japanese, emoji) means a 10000-char instruction can be 30 000+ bytes. Byte-level enforcement is the only correct interpretation.

**Behavior (Phase 1, deferred — described here for forward reference):**

3. Inside the Phase 1 `XVLAAdapterChunkPredictor.predict` (when `ModelRuntime.__call__` is implemented), reuse `src/vla_project/data/transforms/language.py::GemmaPromptTokenizer`. After tokenization, if `len(tokens) > prompt_max_len` (resolved from ckpt `meta.cfg.data.prompt_max_len` with fallback `vla_project.data.constants.DEFAULT_PROMPT_MAX_LEN=20`), silently truncate AND emit `logger.warning({event: prompt_truncated, original_token_count, prompt_max_len})`.

**Rationale:** in Phase 0 the predictor is a stub (HoldPos ignores instruction; xvla_adapter returns 500). Loading the Gemma tokenizer at server startup adds a HF `AutoTokenizer` dependency + a `tokenizer/` directory inside the ckpt export, neither of which has any functional effect until Phase 1's forward pass exists. Pydantic-level byte sanity is sufficient for Phase 0 abuse defense.

### F3. Proprio OOD warn + extreme-value hard reject

**Problem:** `_normalize_proprio` clips silently. A deg-vs-rad swap (proprio sent in degrees but training stats are in rad) produces `|normed| ≈ 30`, all dims clipped to ±1 → model receives zero discriminative signal but operator sees no error. Operators currently debug this by training-loss spike or rollout failure, not at the inference server.

**Behavior:** in `DomainAdapter`:

1. At top of `preprocess`, after `proprio_raw = np.asarray(req.proprio, dtype=np.float32)`:
   ```python
   if not np.isfinite(proprio_raw).all():
       bad_dims = np.where(~np.isfinite(proprio_raw))[0].tolist()
       raise ValueError(f"proprio contains non-finite values at dims {bad_dims}")
   ```
2. In `_normalize_proprio`, *before* clip:
   ```python
   excess = float(np.max(np.abs(normed)) - 1.0)   # |normed| - 1.0
   if excess > (PROPRIO_OOD_HARD_ABS - 1.0):       # |normed| > 10.0
       hard_dims = np.where(np.abs(normed) > PROPRIO_OOD_HARD_ABS)[0].tolist()
       raise ValueError(
           f"proprio normalized |x|>{PROPRIO_OOD_HARD_ABS} at dims {hard_dims} "
           f"(max excess {excess:.2f}); likely unit mismatch "
           f"(deg/rad swap or wrong proprio_key)"
       )
   if excess > (PROPRIO_OOD_WARN_ABS - 1.0):       # |normed| > 1.0 (not raising)
       ood_dims = np.where(np.abs(normed) > PROPRIO_OOD_WARN_ABS)[0].tolist()
       logger.warning(json.dumps({
           "event": "proprio_ood",
           "ood_dim_count": len(ood_dims),
           "ood_max_excess": round(excess, 3),
           "ood_dims": ood_dims,
       }))
   ```
   where `PROPRIO_OOD_WARN_ABS = 1.0` and `PROPRIO_OOD_HARD_ABS = 10.0`.
3. Clip behavior unchanged.

**Note on warn-vs-raise ordering:** the hard-reject branch runs *first* and skips
the OOD warning entirely (raising will produce an `invalid_request` log via the
existing 422 handler — the OOD warn would be a redundant second log line for
the same event). The warn branch only fires for inputs that are absorbed by
clipping and therefore proceed through the control loop.

**Rationale:**

- `>10x` was chosen over the brainstorm's initial `>5x`: ckpt q-range is statistical, not physical, so `5x` can fire on legitimate startup poses outside training support. `10x` is wide enough to admit any realistic pose-from-distribution-tail while still catching the deg/rad case (rad joint ≈ 0.5 rad max → 30 deg → ~30x q-range when sent in degrees).
- NaN/inf is hard-rejected unconditionally — there is no "warn-only" interpretation of `inf` proprio.
- The warn-on-clip log gives operators a structured signal (`event=proprio_ood`) for log searches, without breaking the control loop.

**No yaml control.** Hard limits (`>10x`, NaN/inf reject) are universal sanity, not per-robot tuning.

### F4. Wire-typo guard (preserves `extra="ignore"` forward-compat)

**Problem:** `extra="ignore"` silently drops `image_wirst`, `image_pirmary`, `propio` typos. Operators discover these via "model output is constant" or "wrist appears black" — far downstream of where the typo lives.

**Behavior:** add `model_validator(mode="before")` to `PredictRequest`. The validator iterates the raw incoming dict's keys and applies, in this order, for each key NOT in the known modeled-fields set:

```
MODELED_FIELDS = {
    "image_primary", "image_wrist",
    "proprio", "instruction",
    "model_version", "_t_mono_ns",
}
```

For each unknown `key`:

1. **Near-miss check (uses Damerau-Levenshtein distance ≤ 1):** if `key` is within edit distance 1 of any name in `MODELED_FIELDS`, raise
   `ValueError(f"unknown field {key!r}; did you mean {nearest!r}?")`.
   (Damerau-Levenshtein, not plain Levenshtein, because the typical mistake — `pirmary`, `wirst`, `mnoo` — is a single-character **transposition**, which has plain-Levenshtein distance 2 but Damerau-Levenshtein distance 1.)
2. **Image-prefix check:** if `key` starts with `"image_"` (and didn't trip rule 1), raise
   `ValueError(f"unknown image field {key!r}; known: image_primary, image_wrist")`.
3. Otherwise: silently ignore (`extra="ignore"` semantics preserved for `_request_id`, `_session_token`, `_trace`, future MimicRec observability fields).

**Rationale:** rule 1 produces the helpful `did you mean X?` message for the common cases (`image_pirmary`, `image_wirst`, `propio`, `instructionn`). Rule 2 is the safety net for image-shaped typos that are NOT within distance 1 (e.g., `image_camera_left` if a future ckpt accidentally requests it without alias remap). Rules 1+2 together still preserve forward-compat with arbitrary `_X` metadata.

**Implementation note:** Damerau-Levenshtein with bound 1 is ~15 lines of stdlib Python (no `python-Levenshtein` dep). The `MODELED_FIELDS` set is small (6) and the typical request has ~5 keys, so the validator is O(30) string comparisons per request — negligible against the 266 ms latency budget.

### F5. `/admin/schema` introspection endpoint

**Problem:** clients (MimicRec contract YAML authors, debug scripts, future internal tools) currently have to open the deploy yaml + ckpt `meta.json` to learn what the server expects. There is no machine-readable handshake.

**Behavior:** new `GET /admin/schema` route in `inference_server.py`. No auth (server binds 127.0.0.1 by default in Phase 0). Returns JSON:

```json
{
  "predictor": "XVLAAdapterChunkPredictor",      // or HoldPositionChunkPredictor
  "ckpt": {
    "expected_unnorm_key": "libero_spatial_no_noops",
    "expected_action_chunk_len": 8,
    "expected_action_dim": 7,
    "expected_proprio_dim": 8
  },
  "wrist_hard_required": false,
  "request_fields": {
    "scene_image": "image_primary",
    "wrist_image": "image_wrist",
    "proprio": "proprio",
    "instruction": "instruction"
  },
  "proprio": {
    "source": {
      "components": [
        {"name": "joint_pos", "dims": 7, "units": "rad"},
        {"name": "gripper_pos", "dims": 1, "units": "normalized_0_1"}
      ],
      "total_dim": 8
    }
  },
  "image": {"min_side": 64, "max_side": 4096},
  "instruction": {"max_bytes": 10000},
  "prompt": {"max_tokens": null},                // null in BOTH modes for Phase 0 (see below)
  "proprio_ood": {"warn_threshold": 1.0, "hard_threshold": 10.0}
}
```

**`prompt.max_tokens` in Phase 0 is always `null`.** The server does not tokenize or truncate yet (per F2 deferral), so reporting a `max_tokens` value would imply an enforced constraint that does not exist. Phase 1 will populate the field from `runtime.cfg.data.prompt_max_len` (fallback `vla_project.data.constants.DEFAULT_PROMPT_MAX_LEN=20`) at the same time the tokenize-truncate-warn path is implemented in `XVLAAdapterChunkPredictor.predict`. Returning `null` in both Phase 0 modes (instead of just `hold_position`) makes the schema's truthfulness symmetric across predictors and avoids clients writing client-side validators against an unenforced ceiling.

**Excluded fields and why:**
- `runtime.device`, `runtime.dtype`, `runtime.torch_compile`, `runtime.warmup_iters` — not the client's business.
- `step` — debug only; clients build requests against shape, not training step.
- `action.native`, `action.contract`, `action.frame_conversion`, `action.denormalization` — the truth is the deploy yaml + the MimicRec pairing yaml. Re-exporting it from the server adds drift surface without payoff (clients already have the pairing yaml on disk).

---

## Section 2. Component changes

### Modified files

#### `src/vla_project/deployment/schemas.py`

```python
# config: keep extra="ignore" (forward-compat with MimicRec observability fields)
# add: byte-length validator on instruction (UTF-8, max 10000 bytes; empty allowed)
# add: model_validator(mode="before") for typo guard (image-prefix + edit-distance)
```

No new fields. `instruction: str` keeps its existing type; `populate_by_name=True` and `_t_mono_ns` alias unchanged.

#### `src/vla_project/deployment/domain_adapter.py`

1. `_decode_jpeg_b64`: split into header-parse (size check) + body-decode (`convert("RGB")`).
2. `preprocess`: top-level `np.isfinite(proprio_raw).all()` assert.
3. `_normalize_proprio`: pre-clip excess computation + warn / hard reject.

No constructor signature change (no `tokenizer` / `prompt_max_len` injection — deferred per F2).

#### `src/vla_project/deployment/inference_server.py`

1. New `/admin/schema` GET route. Returns the JSON shape from F5 with `prompt.max_tokens=None` for both predictor modes (Phase 0 deferral).
2. No tokenizer instantiation (deferred per F2).
3. No `prompt_max_tokens` resolution either (deferred to Phase 1 alongside actual tokenization).

#### Module-level constants (per codex, no new `constants.py` for this add-on)

Constants live next to their use site (one declaration per relevant module):

- `domain_adapter.py`: `IMAGE_MIN_SIDE = 64`, `IMAGE_MAX_SIDE = 4096`, `PROPRIO_OOD_WARN_ABS = 1.0`, `PROPRIO_OOD_HARD_ABS = 10.0`.
- `schemas.py`: `INSTRUCTION_MAX_BYTES = 10_000`.

### Unchanged files

- `predictors/{base,hold_position,xvla_adapter}.py`
- `runtime.py`
- `configs/deploy/*.yaml`
- New `prompt_processor.py` is **not** created (the deferred Phase 1 path will reuse `data/transforms/language.py::GemmaPromptTokenizer` directly, per codex's anti-duplication note).

---

## Section 3. Testing strategy

### New test files (5)

| File | Coverage |
|---|---|
| `tests/test_validation_image_sanity.py` | **Unit-level (`_decode_jpeg_b64` direct call):** 32×32 → `ValueError` / 5000×5000 → `ValueError` / 64×64 boundary → ok / 4096×4096 boundary → ok / 4097×4097 → `ValueError`. **Integration-level (FastAPI TestClient):** one 32×32 → 422 case + one 224×224 → 200 case (full path coverage; large-resolution boundary cases stay as unit tests to avoid heavy round-trips). |
| `tests/test_validation_prompt.py` | `""` → 200 / "hello" → 200 / 10 000-byte ASCII → 200 / 10 001-byte ASCII → 422 / multi-byte UTF-8 (Japanese 5 000 chars ≈ 15 000 bytes) → 422 / 9 999-byte ASCII (boundary) → 200. |
| `tests/test_validation_proprio.py` | NaN at dim 3 → 422 (msg names dim) / inf at dim 7 → 422 / `\|normed\|=1.5` → 200 + caplog WARNING with `event=proprio_ood` (clip absorbs) / `\|normed\|=11.0` (deg-rad swap simulation) → 422 with msg containing "unit mismatch" and **no** `proprio_ood` warn (hard reject skips the warn per F3 ordering) / `\|normed\|=10.0` boundary → 200 + warn / `\|normed\|=10.01` → 422. |
| `tests/test_validation_typo.py` | `image_pirmary` → 422 ("did you mean 'image_primary'") / `image_wirst` → 422 ("did you mean 'image_wrist'") / `propio` → 422 ("did you mean 'proprio'") / `_t_mono_n` → 422 ("did you mean '_t_mono_ns'") / `image_camera_left` (no near miss) → 422 ("unknown image field; known: image_primary, image_wrist") / `_request_id` → 200 (silently ignored) / `_trace` → 200 / `model_versionn` → 422 ("did you mean 'model_version'"). |
| `tests/test_admin_schema.py` | `GET /admin/schema` in `xvla_adapter` mode returns the 9 top-level keys (`predictor`, `ckpt`, `wrist_hard_required`, `request_fields`, `proprio`, `image`, `instruction`, `prompt`, `proprio_ood`) with `prompt.max_tokens=null` (Phase 0 deferral) / in `hold_position` mode same shape with `prompt.max_tokens=null` / response is valid JSON / `expected_*` fields match deploy yaml + meta.json. |

### Extended tests

- `tests/test_domain_adapter.py`: add proprio-NaN case (1 line) + image-bound case via `_decode_jpeg_b64` direct call (3 lines). Existing tests do not break (constructor signature unchanged).
- `tests/test_serve_smoke.py`: add 1 typo case (`image_pirmary` → 422) to verify FastAPI integration end-to-end.
- `tests/test_inference_server_minimal.py`: assert `/admin/schema` exists alongside `/healthz`.

### Phase 0 acceptance gate addition

Existing block in `src/vla_project/deployment/README.md`:

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

Append the 5 new test files to that command.

### MimicRec smoke (Phase 0 acceptance gate item 3)

Existing `scripts/smoke_inference_real_data.py` flow uses canonical wire field names + reasonable image sizes + finite proprio + empty/short instruction. **No regression expected.** Verification step in plan: run the existing smoke and confirm it still passes with no code changes on the MimicRec side.

---

## Section 4. Error handling summary

| Failure | Exception | HTTP | Log outcome | Notes |
|---|---|---|---|---|
| typo guard | `ValueError` (validator) → `RequestValidationError` | 422 | `invalid_request` | message includes `did you mean X?` when edit distance hits |
| instruction > 10 000 bytes | `ValueError` (validator) → `RequestValidationError` | 422 | `invalid_request` | byte count in msg |
| proprio NaN/inf | `ValueError` | 422 | `invalid_request` | dim list in msg |
| image side out of bound | `ValueError` | 422 | `invalid_request` | `(W, H)` in msg |
| proprio `\|normed\| > 10` | `ValueError` | 422 | `invalid_request` (single log line) | dim + excess in msg, hint about unit mismatch. **No** `proprio_ood` WARNING (per F3 ordering: hard reject skips warn) |
| proprio `\|normed\| > 1` (and ≤ 10) | (none) | 200 | `ok` (control flow) + separate `WARNING` log with `event=proprio_ood` | structured for log search; clip absorbs the value |
| Phase 1 tokenize truncation | (none) | 200 | `ok` + separate `WARNING` log with `event=prompt_truncated` | not in Phase 0 |

All 422 paths feed the existing `_log_request(... outcome="invalid_request" ...)` channel.

---

## Section 5. Out of scope (Phase 1 work)

- **Tokenize-time truncation + warn log (F2 step 3).** Deferred until `XVLAAdapterChunkPredictor.predict()` is implemented. At that point reuse `data/transforms/language.py::GemmaPromptTokenizer`; no new module.
- **Multi-view image schema (3+ cameras).** Phase 1 alias remap (per Phase 0 spec §Section 5).
- **`Image.MAX_IMAGE_PIXELS` global override.** Header-parse-first (F1 step 1) bounds the worst case for plausible attackers; explicit `MAX_IMAGE_PIXELS` enforcement is Phase 1 hardening if/when the server is exposed beyond LAN.
- **Per-robot proprio physical bounds.** Would belong in deploy yaml's `proprio.source` (or a new `proprio.physical_bounds`) and would conflict with the architectural principle. Defer until a concrete robot demands it.
- **`model_version` policy.** Today: declared in schema, server ignores it. Phase 1 may add `logger.warning` on mismatch with `runtime.cfg.export.model_version`, but the rejection policy stays "never reject" until a concrete need arises.
- **Request body size limit at ASGI layer.** sysadmin / deployment concern, not server-code concern.

---

## Section 6. Known limitations

- Typo detection uses bounded Damerau-Levenshtein distance ≤ 1 against the `MODELED_FIELDS` set (catches single substitutions, insertions, deletions, and adjacent transpositions). It will miss compound typos (`image_priary_wrist` — distance 2+) and false-positive on intentional new fields exactly one edit away (e.g., a hypothetical future `image_primarys` would fire). The image-prefix fallback rule provides a stricter overlay for image fields specifically; non-image typos that don't trip the near-miss guard fall through to `extra="ignore"` and re-surface as functional bugs (e.g., a missing proprio causes the existing `proprio length 0 != ...` 422). Acceptable trade-off for Phase 0.
- `/admin/schema` exposes ckpt-derived facts that change on ckpt swap. Clients that cache the response will see stale info if they don't re-fetch on server restart. Phase 0 documents this; no `ETag` / version field. Phase 1 may add `step` to the response for cache invalidation.
- Proprio OOD warn fires on every clipped sample. In a steady-state deployment with mild distribution drift this can be noisy. Operators are expected to monitor `event=proprio_ood` aggregate counts, not individual lines. If noise becomes a problem, F3 step 2 can be extended to rate-limit the warn (e.g., one log per 100 occurrences); deferred until measured noise actually hurts.

---

## Acceptance checklist

- [ ] `schemas.py` keeps `extra="ignore"`; adds `instruction` byte validator + typo `model_validator`.
- [ ] `domain_adapter.py` rejects NaN/inf proprio at preprocess entry.
- [ ] `domain_adapter.py` checks image side bounds before pixel decode.
- [ ] `domain_adapter.py` `_normalize_proprio` warn-logs at `>1` excess and raises at `>10`.
- [ ] `inference_server.py` serves `GET /admin/schema` with `prompt.max_tokens=null` in both modes (Phase 0 deferral).
- [ ] All 5 new test files green; 3 extended tests still green; existing 8 acceptance tests still green.
- [ ] MimicRec smoke (`smoke_inference_real_data.py`) still green with no client-side change.
- [ ] No new `configs/deploy/*.yaml` fields.
- [ ] No new `predictors/*.py` constructor changes.
- [ ] `tools/export_checkpoint.py` is NOT introduced as part of this change (tokenizer load remains Phase 1).
