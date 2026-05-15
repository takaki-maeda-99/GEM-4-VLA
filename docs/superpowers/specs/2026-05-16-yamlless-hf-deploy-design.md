# Yaml-less HF-driven Deployment Server — Design Spec

**Date**: 2026-05-16
**Status**: Draft, awaiting user approval
**Supersedes**: portions of [2026-05-06-vla-inference-server-design.md](2026-05-06-vla-inference-server-design.md) (DeployConfig schema, DomainAdapter contract translation)
**Predecessor PR**: c0b7dac (HF id resolution in `ModelRuntime.from_export`)

## 1. Problem

The inference server (`scripts/serve.py` + `src/vla_project/deployment/`) requires both `--checkpoint <ckpt>` and `--deploy-config <yaml>`. The yaml encodes:

1. Asserts against `meta.json` (redundant — same data is in the ckpt).
2. Client raw-proprio layout and per-step transformation pipeline (deg_to_rad, copy, pad_zeros) into model-input shape.
3. Wire field-name aliasing (`request_fields.scene_image: image_primary` etc.).
4. Client contract: expected action frame, gripper convention.
5. Frame conversion method (currently only `none` is implemented).
6. Smoke-test escape hatch (`wire_only_smoke`).
7. Runtime knobs (device, dtype, torch_compile, warmup).

Items 1, 3 (when canonical), 6 (default false), and 7 (CLI flags exist) are redundant or pure noise. Items 2 and 4 are client-side facts that depend on the consumer's robot — they are NOT properties of the model checkpoint. Item 5 is partially dead: the only conversion path `world_to_ee_local`/`ee_local_to_world` is not implemented, and all existing yamls run with `wire_only_smoke=true` to skip the corresponding asserts.

The result: every new model author must write a ~100-line yaml that mostly re-asserts what `meta.json` already says, while the *load-bearing* fields describe the **client robot** and are therefore wrong to ship alongside the model.

## 2. Decision

Move contract translation **out of the server** and into the client. The server returns **model-native, fully q99-denormalized action chunks** (with optional per-checkpoint `post_process.py` applied). The yaml is deleted. The server takes only `--checkpoint <local|HF-id>` and runtime knobs.

The architectural boundary becomes:

```
client  →  raw bytes (canonical PredictRequest schema)
                ↓
server  →  q99-normalize proprio → forward → q99-denorm action chunk → post_process.apply
                ↓
client  ←  actions in model NATIVE units (frame, gripper convention per meta.native_action)
                ↓
client  →  contract translation (frame conversion, gripper convention, robot send)
```

The server is **model-faithful**: it does not pretend to translate to any particular robot.

## 3. CLI shape

```bash
uv run python scripts/serve.py \
  --checkpoint <local-dir | HF-id | HF-id/subfolder> \
  [--predictor xvla_adapter | hold_position]   # default: xvla_adapter
  [--domain-id N]                              # default: cfg.data.domain_id
  [--unnorm-key KEY]                           # required iff norm_stats has >1 keys
  [--trust-checkpoint-code]                    # required to load post_process.py from HF-resolved ckpts; see §6
  [--host 127.0.0.1] [--port 8001]
  [--device cuda:0] [--dtype bf16]
  [--torch-compile off] [--warmup-iters 1]
```

Removed:
- `--deploy-config` (the central change).

Changed:
- `--predictor`: was `required=True`, now optional with `xvla_adapter` as default. `hold_position` remains for non-GPU wire-protocol smoke tests; in that mode, `--checkpoint` becomes optional (`hold_position` will derive chunk_len and action_dim from `--checkpoint` if given, otherwise require `--action-dim N --action-chunk-len M`).

## 4. Server endpoints

### `POST /predict`

Request body (`PredictRequest`, schema unchanged from current code):
```
{
  "image_primary": "<b64 JPEG>",
  "image_wrist":   "<b64 JPEG>" | null,
  "proprio":       [float, ...]   # MODEL-input shape, NOT raw client format
  "instruction":   "<str>"
}
```

The `proprio` field is the **model-input dimensionality already adapted by the client** (e.g., 8-dim for the bottle ckpt, with `[ee_pos(3), ee_rotvec(3), gripper(1), 0_pad(1)]` layout per the model's training distribution). The server does NOT perform `deg_to_rad`, `copy`, `pad_zeros`, or any reshape — the burden moves to the client.

Response body:
```
{"actions": [[float, ...], ...]}    # shape (T, A) in MODEL native units
```

The `actions` are:
- q99-denormalized on `norm_stats[unnorm_key].action.mask == True` dimensions
- passthrough on `mask == False` dimensions
- transformed by `post_process.apply(actions, meta)` if a `post_process.py` is bundled with the ckpt

### `GET /metadata`

```
{
  "step":                  int,         # meta.step
  "model_name":            str,         # cfg.language.model_name
  "git_commit":            str,         # meta.git_commit
  "action_dim":            int,         # len(norm_stats[key].action.mean)
  "proprio_dim":           int,         # len(norm_stats[key].proprio.mean)
  "action_chunk_len":      int,         # cfg.data.action_chunk_len
  "domain_id":             int,         # active server domain
  "num_domains":           int,         # cfg.model.num_domains
  "unnorm_key":            str,         # active key
  "native_action":         {...} | null, # meta.native_action (see §5)
  "has_post_process":      bool,
  "post_process_module":   str | null    # absolute path for ops visibility
}
```

Clients self-configure proprio shape and action dim from this endpoint at startup.

### `GET /healthz`

Unchanged from current implementation.

## 5. `meta.native_action` schema (new)

New top-level field in `meta.json` written by `training/checkpoint.py`:

```json
"native_action": {
  "units": "meter_axisangle_rad",
  "frame": "world" | "ee_local",
  "gripper": {
    "kind":  "absolute" | "delta" | "binary",
    "units": "normalized_0_1" | "signed_neg1_pos1" | "binary_threshold_0p5",
    "sign":  {"closed": <float>, "open": <float>}
  }
}
```

Source: training-side `configs/train/*.yaml` gains an optional `data.native_action: {...}` block. If declared, `checkpoint.py` writes it into `meta.json`. If absent at training time, the checkpoint's `meta.native_action` is omitted entirely.

Server behavior when `meta.native_action` is absent:
- Startup logs `WARNING: native_action metadata absent; clients must know action convention out-of-band`.
- `/metadata` returns `"native_action": null`.
- Startup does NOT fail. (Real-robot clients should gate on `native_action != null`.)

Backfill of existing HF checkpoints is handled by a one-off tool — see §7 (Add).

## 6. `post_process.py` convention

### File location

`<ckpt_dir>/post_process.py` (next to `meta.json`).

### Required interface

```python
import numpy as np

def apply(actions: np.ndarray, meta: dict) -> np.ndarray:
    """
    actions: shape (T, A), post q99-denorm, model native units.
    meta:    the loaded meta.json dict (read-only for post_process).
    Returns: same shape (T, A).
    """
```

### Loader

```python
import sys, importlib

def load_post_process(
    ckpt_dir: Path,
    *,
    is_local: bool,
    trust_checkpoint_code: bool,
) -> Callable | None:
    pp_file = ckpt_dir / "post_process.py"
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
            del sys.modules["post_process"]   # force re-import for test isolation
        mod = importlib.import_module("post_process")
        fn = getattr(mod, "apply", None)
        if not callable(fn):
            raise HardFailAssertion(f"{pp_file} missing callable apply(actions, meta)")
        logger.warning(
            f"loaded executable post_process from ckpt: {pp_file}. "
            f"This file runs with full server privileges."
        )
        return fn
    finally:
        sys.path.pop(0)
```

`is_local` is supplied by `ModelRuntime` based on which branch of `_resolve_ckpt_dir` was taken: `True` if the input was a local existing path, `False` if it triggered `snapshot_download`.

`post_process.py` may `import gripper_normalizer` (and other sibling modules in the ckpt dir) directly — sibling files are on `sys.path` during import. Relative imports (`from .X import ...`) are NOT supported (the ckpt dir is not a Python package).

### Trust model

Loading `post_process.py` runs arbitrary Python with full server privileges. **This IS a new RCE surface for this repo**: `model.pt` is loaded via `torch.load(..., weights_only=True)` in `training/checkpoint.py:109`, so today there is no code-execution path through ckpt artifacts. Adding `post_process.py` import without gating would silently enable arbitrary code execution any time a user pulls an HF checkpoint.

Policy:

- **Local ckpt path** (`--checkpoint /local/dir`, or any path that exists on disk): load `post_process.py` by default with a startup `WARNING` log. Local paths are owned by the user; no opt-in needed.
- **HF-resolved ckpt** (`--checkpoint org/repo[/subfolder]` triggered the `snapshot_download` path in `runtime._resolve_ckpt_dir`): the loader refuses to load `post_process.py` UNLESS `--trust-checkpoint-code` is explicitly passed on the CLI. Without the flag, the server still starts but logs `WARNING: <ckpt>/post_process.py present but skipped (pass --trust-checkpoint-code to enable)`, and actions are returned without post-processing.

The `--trust-checkpoint-code` flag is the explicit opt-in. It must be added to `scripts/serve.py` and propagated through `inference_server.build_app` to the post-process loader. The loader is given a `is_local: bool` argument by `ModelRuntime` based on which branch of `_resolve_ckpt_dir` was taken; the policy above lives in the loader.

This policy avoids two failure modes:
1. A user fetches an HF ckpt expecting actions to be normalized (e.g., bottle gripper) but forgets the flag — they get raw passthrough actions instead of silent compromise. The WARN log is loud enough that the failure mode is observable.
2. A malicious or compromised HF repo cannot inject code without the user explicitly opting in.

### Failure modes

| Failure | When | Effect |
|---|---|---|
| HF-resolved ckpt + no `--trust-checkpoint-code` + `post_process.py` present | startup | Skip load, emit `WARNING`. Actions served raw (no post-process). Server starts. |
| `post_process.py` syntax error / ImportError | startup | `HardFailAssertion`, server does not start |
| `apply` not defined or not callable | startup | `HardFailAssertion` |
| `apply` raises at runtime | per-request | 500 `postprocess_error` (existing flow) |
| `apply` returns wrong shape / NaN | per-request | 500 `postprocess_error` (existing NaN+shape guard) |

## 7. Components — delete / move / add

### Delete

- `configs/deploy/{_template,so101_v46,v36_libero_spatial,mimicrec_pairing_example}.yaml`
- `src/vla_project/deployment/domain_adapter.py:DeployConfig` (lines 55-165) and all sub-models
- `src/vla_project/deployment/domain_adapter.py:load_deploy_config`
- `src/vla_project/deployment/domain_adapter.py:DomainAdapter` class — preprocess (proprio source/adapt/field-aliasing parts), postprocess (frame conversion, gripper conversion), `_q99_denorm_action` is salvaged (see Move)
- `validate_startup_xvla` contract checks (lines 432-461): frame mismatch, gripper degeneracy, deploy.proprio.adapt.output_dim assertion
- `validate_startup_hold_position` contract checks (lines 482+, scoped review during impl)
- Constants tied to deploy yaml: `wire_only_smoke` paths
- `scripts/serve.py`: `--deploy-config` argument

### Move

`domain_adapter.py` is rewritten as smaller focused modules:

- `src/vla_project/deployment/wire_io.py` — JPEG decode + image transform integration + proprio normalize + q99 denorm with mask. Pulls `_q99_denorm_action`, F1 image bounds, F3 OOD proprio + isfinite, NaN guards.
- `src/vla_project/deployment/post_process_loader.py` — §6 loader.
- `src/vla_project/deployment/metadata.py` — `/metadata` response builder.
- `src/vla_project/deployment/startup_validation.py` — non-contract subset of `validate_startup_xvla` (see §8).

### Add

- `meta.native_action` schema (training side) — modify `src/vla_project/training/checkpoint.py` to include the block if present in `cfg.data.native_action`.
- `tools/backfill_meta_native_action.py` — one-off backfill tool for existing HF checkpoints. Accepts `--ckpt <local|HF> --units --frame --gripper-kind --gripper-units --gripper-closed --gripper-open`, rewrites local meta.json. HF push is manual.
- `tests/deployment/` — six new tests per §9. Existing tests that exercise DomainAdapter / DeployConfig are deleted.

## 8. Startup validation (post-refactor)

Single function in `startup_validation.py`. Receives `meta` (loaded meta.json), `runtime` (`ModelRuntime` instance), `args` (parsed CLI). Performs:

1. `domain_id ∈ [0, cfg.model.num_domains)` — preserves `domain_adapter.py:398-402`.
2. `unnorm_key ∈ meta.norm_stats` — new explicit check (currently relies on `KeyError` at first dereference).
3. `cfg.data.action_chunk_len == cfg.model.action_chunk_len` (where the latter exists) — preserves the fallback chain check at lines 408-418.
4. `len(norm_stats[key].action.mean) == cfg.model.action_dim` (where `cfg.model.action_dim` is the resolved output dim of the action head).
5. `len(norm_stats[key].proprio.mean) == cfg.model.proprio_dim`.
6. **NEW**: `q01/q99/mask/std/min/max` shapes all agree per `action` and `proprio`. (Currently only `mean` is length-checked.)
7. Wrist hard-required derivation from `cfg.model.{use_wrist_bridge, use_scene_wrist_dinov2_llm, wrist_in_llm + wrist_view_dropout_p}` — preserves lines 462-480, but the check fires on PredictRequest schema (the canonical fields) rather than yaml `request_fields`.
8. `post_process.py`, if present, imports cleanly and exports a callable `apply`.
9. `meta.native_action` absent → `WARNING` log, do not fail.

Removed (contract):
- Frame mismatch + `frame_conversion=none` (line 440-449)
- Gripper sign degeneracy (line 450-461)
- `cfg.proprio.adapt.output_dim == expected_proprio_dim` (line 432-436)
- `wire_only_smoke` escape hatch (entire concept)

## 9. Tests

New tests under `tests/deployment/`:

1. **`test_serve_yamlless_smoke.py`** — integration. Boot server with `--checkpoint takaki99/GEM-4-FT-bottle`. Hit `/metadata`. Send canonical PredictRequest with bottle-shape proprio (8 floats). Assert response `actions` shape `(8, 7)` and `gripper` dim is in `[0, 1]` (post_process applied).
2. **`test_post_process_loader.py`** — unit. Cases: (a) local + file valid → returns callable; (b) local + file absent → returns None; (c) local + file but no `apply` → `HardFailAssertion`; (d) local + file but ImportError → `HardFailAssertion`; (e) HF-resolved (`is_local=False`) + file present + flag off → returns None with WARN; (f) HF-resolved + file present + flag on → returns callable. Each case asserts the documented behavior.
3. **`test_metadata_endpoint.py`** — unit + integration. All fields present for a bottle-style ckpt; `native_action: null` for an old ckpt without the block.
4. **`test_denorm_with_mask.py`** — unit. Build synthetic norm_stats with `mask = [T,T,T,T,T,T,F]`, run `wire_io.q99_denorm`. Verify mask=True dims are q99 inverse and mask=False dim is passthrough.
5. **`test_startup_validation.py`** — unit. Each of `domain_id` out of range, `unnorm_key` missing, `q01/q99` shape mismatch, `cfg.model.action_dim` ≠ stats dim must `HardFailAssertion` at boot.
6. **`test_backfill_native_action.py`** — unit. Tool reads, rewrites meta.json, idempotent on second run.

Deleted tests (DomainAdapter / DeployConfig dependencies):
- Any test under `tests/deployment/` that imports `DomainAdapter`, `DeployConfig`, or `load_deploy_config`. Exact list determined at implementation time.

## 10. Migration & backwards compatibility

- This is a **breaking change** to the deployment surface.
- The repo is private; the inference server is not a published API. No deprecation path is offered.
- Existing HF checkpoints (`takaki99/GEM-4-FT-bottle`, `takaki99/so101-v46`) work with the new server immediately, EXCEPT they lack `meta.native_action`. The startup `WARNING` fires.
- The bottle ckpt needs a `post_process.py` shim added on HF (~5 lines, wraps `gripper_normalizer.normalize_gripper`) for the gripper to land in `[0, 1]`. Without it, the server returns `actions[..., 6]` as raw `gripper_pos / 100` (the model's training-time mask=False passthrough form).
- Future model authors: ship `meta.json` + `model.pt` (+ optional `post_process.py` + optional sibling modules) to HF, write `data.native_action` in their training yaml. Nothing else.
- PR description must state: "contract translation (frame conversion, gripper convention mapping, raw proprio adaptation) is moved to clients. `configs/deploy/*.yaml` removed. Clients must send model-input proprio and interpret model-native action chunks."

### Proprio layout discoverability

`/metadata` exposes `proprio_dim` (a single integer) but does NOT expose the **semantic layout** of those dimensions (e.g., "[ee_pos(3), ee_rotvec(3), gripper(1), 0_pad(1)]"). Clients must derive layout from the training config / dataset convention out-of-band. Adding `proprio_layout` to `/metadata` is YAGNI for the current single-author / single-consumer reality but is the natural extension point if the consumer base grows.

## 11. Out of scope

- Per-request `domain_id` field. Server-startup `--domain-id` is sufficient for current FT use case. Internal validation (`0 <= id < num_domains`) is structured so per-request switching can be added later without rewriting.
- A client-side translation library that replicates the deleted contract pipeline (mimicrec adapter, etc.). The server design only commits to "client deals with contract." Building that library is a separate scope.
- Multi-checkpoint serving (serve N models from one process). Not needed; one process per ckpt.
- Declarative `meta.post_process: [{op: ...}]` manifest. Considered, rejected: the bottle gripper case already needs code, premature constraint.
- Public API stability commitments. Internal-only for now.

## 12. Open questions / risks

- **`hold_position` predictor scope**: the existing stub uses `DeployConfig` for action_dim/chunk_len. After refactor, it must either derive these from `--checkpoint` (defeats its no-GPU purpose) or take explicit `--action-dim N --action-chunk-len M`. Implementation defaults to "if `--predictor hold_position` and no `--checkpoint`, require `--action-dim` and `--action-chunk-len` flags."
- **post_process module re-import on hot reload**: out of scope; server processes are single-shot per ckpt.
- **`native_action` schema versioning**: no version field for now. If we later need to change schema, a `meta.schema_version` is the natural place — but adding it now is YAGNI.
