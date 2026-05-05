# VLA Inference HTTP Server — Design

**Status:** ready for user review (all 6 sections drafted, codex peer review converged)
**Date:** 2026-05-06
**Owner:** Takaki Maeda

## Context

MimicRec is a robot teleoperation/replay client that drives a real arm at 15–30 Hz via a closed-loop controller. It calls a VLA inference server over HTTP, sending per-tick observations and receiving short action chunks. The wire contract (request/response JSON, units, frame, latency budget, failure semantics) is defined by MimicRec and is **not** overridable from the server side — the server must absorb every robot-specific quirk.

This document specifies the HTTP server that hosts an X-VLA-Adapter checkpoint and answers MimicRec's `POST /predict`. Phase 0 targets the v36 architecture (LIBERO Spatial single-domain, `wrist_in_llm` + `wrist_view_dropout_p=0.3`, `num_domains=1`); the v36 ckpt itself is produced via a short training run on `configs/train/libero_spatial_v36.yaml` once the server skeleton exists. The architecture is parametric so future checkpoints (multi-domain pre-trained, SO-101-fine-tuned) plug into the same machinery via deploy yamls.

### Hard constraints (from MimicRec)

- One route: `POST /predict` (configurable path).
- Latency target ~266 ms per call (chunk_size=8 @ 15 fps × 0.5 prefetch threshold).
- Action layout per row: `[Δx, Δy, Δz, Δrx, Δry, Δrz, gripper]` (6 ee_delta + 1 gripper).
- Pose units: `meter_axisangle_rad` (the only mode in MVP).
- `max_inflight=1` — server may serialize.
- No retry on failure → server must aim for ≥99% success in steady state.

### Decisions locked

| # | Decision |
|---|---|
| 1 | Cross-embodiment is a **pre-training** concern (multi-domain when trained that way). At deploy time, one process drives **one** robot. v36 specifically is single-domain (`num_domains=1`); future multi-domain pre-trained ckpts use the same machinery via `--domain-id`. |
| 2 | `domain_id` is a **server-startup argument**, not a wire field. One process = one (ckpt, domain_id, deploy_config) tuple. |
| 3 | `extra_fields.model_version` is informational only; server does not dispatch on it. |
| 4 | Per-domain "absorption" config split: numeric stats live in **ckpt metadata**; MimicRec field-name mapping / gripper convention / frame live in `configs/deploy/<domain>.yaml`. |
| 5 | Phase 0 ships the skeleton + `HoldPositionChunkPredictor` first; `XVLAAdapterChunkPredictor` plugs in once a v36 ckpt is produced. |
| 6 | Architecture is monolithic FastAPI process (Arch 1). No `/admin/reload`. Restart-on-swap. |

---

## Section 1. Overview

**Purpose:** host an X-VLA-Adapter checkpoint behind `POST /predict`, return action chunks shaped to whatever MimicRec contract the deployed robot uses.

**Top-level layering:**

```
[HTTP layer]            FastAPI route + pydantic schemas (wire shape, contract field names)
        │
[DomainAdapter]         per-domain in/out conversion (proprio shape/units, gripper convention,
        │               frame, action denormalization, MimicRec field-name mapping)
        │
[ChunkPredictor]        ABC. Concrete: HoldPosition / XVLAAdapter[v33/v35/v36→…]
        │               input = internal batch dict, output = native [T, A] np.ndarray
        │
[ModelRuntime]          torch model + tokenizer + image transform + bf16 + (optional) compile
                        Wraps the same primitives policies/xvla_adapter_policy.py uses;
                        differs only in returning the whole chunk (no buffer) and being
                        parametric over gripper/frame conversion (driven by deploy yaml).
```

**Operational model (1 process = 1 unit):**

- Args: `--predictor {hold_position|xvla_adapter} --deploy-config configs/deploy/<domain>.yaml --domain-id <int> --port <N> [--checkpoint <export_dir>]`. `--checkpoint` is required when `--predictor xvla_adapter`, omitted (or unused) when `--predictor hold_position`.
- Startup (depends on `--predictor`):
  - `xvla_adapter`: load ckpt → validate metadata against deploy yaml → warmup forward × 1 (KV cache + JIT stabilize) → ready.
  - `hold_position`: skip ckpt load + ckpt-derived asserts; validate deploy yaml internally consistent (proprio.adapt total dims, action_dim, action_chunk_len) → ready immediately (no warmup needed).
- Stop / swap: kill process and bring up a new one. No reload endpoint.

**Repo placement** (per CLAUDE.md "Recommended Repository Layout"):

- `src/vla_project/deployment/` — new module group.
- `scripts/serve.py` — entry point (thin argparse → uvicorn.run).
- `configs/deploy/` — YAML files, one per (robot, model) combo.

---

## Section 2. Module boundaries

```
src/vla_project/deployment/
├── __init__.py
├── inference_server.py        # FastAPI app + uvicorn entry helper
├── schemas.py                 # pydantic PredictRequest / PredictResponse
├── domain_adapter.py          # DomainAdapter: deploy yaml + ckpt meta → in/out 変換
├── runtime.py                 # ModelRuntime: ckpt load + compile + warmup + device 管理
└── predictors/
    ├── __init__.py
    ├── base.py                # ChunkPredictor ABC (predict(batch) -> np.ndarray[T, A])
    ├── hold_position.py       # HoldPositionChunkPredictor (no model)
    └── xvla_adapter.py        # XVLAAdapterChunkPredictor (v33/v35/v36 wrapper)

scripts/
└── serve.py                   # argparse → uvicorn.run(app)

configs/deploy/
├── _template.yaml             # commented schema reference
└── v36_libero_spatial.yaml    # first concrete deploy (LIBERO Franka, domain_id=0; ckpt produced by training configs/train/libero_spatial_v36.yaml)

tests/
├── test_deployment_schemas.py        # pydantic round-trip + 422 cases
├── test_domain_adapter.py            # proprio.adapt ops, gripper conv, row-shape assert
├── test_predictor_holdposition.py    # zero ee_delta + midpoint gripper
├── test_predictor_xvla_adapter.py    # batch dict assembly + DINOv2 conditional + wrist_mask
├── test_runtime_load.py              # meta.json parse + startup hard-fail asserts
└── test_serve_smoke.py               # FastAPI TestClient end-to-end via HoldPosition
```

**Relationship to existing code:**

- `src/vla_project/policies/xvla_adapter_policy.py` stays as-is — it serves LIBERO sim rollout (`select_action` pops one action at a time, which the rollout loop expects).
- `XVLAAdapterChunkPredictor` (new) is a **separate class**: same model / tokenizer / image transform / Q99 stats primitives, but emits the whole chunk (no buffer), and gripper/frame conversion is **driven by deploy yaml** rather than hardcoded LIBERO conventions.
- Shared primitives (image preprocess, batch build) initially live inline in `xvla_adapter.py`; if duplication becomes painful we extract later. YAGNI.

**`scripts/serve.py` skeleton** (CLAUDE.md "Scripts" — entrypoints stay thin):

```python
import argparse, uvicorn
from vla_project.deployment.inference_server import build_app

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictor", choices=["hold_position", "xvla_adapter"], required=True)
    ap.add_argument("--checkpoint", required=False, default=None)   # required iff --predictor xvla_adapter
    ap.add_argument("--deploy-config", required=True)
    ap.add_argument("--domain-id", type=int, required=True)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--inject-sleep", type=float, default=0.0)      # test-only; injects time.sleep before predict for slow-path smoke
    args = ap.parse_args()
    if args.predictor == "xvla_adapter" and args.checkpoint is None:
        ap.error("--checkpoint required when --predictor xvla_adapter")
    app = build_app(
        predictor_kind=args.predictor,
        checkpoint=args.checkpoint,
        deploy_config_path=args.deploy_config,
        domain_id=args.domain_id,
        inject_sleep_s=args.inject_sleep,
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")

if __name__ == "__main__":
    main()
```

---

## Section 3. Per-request data flow

### Request → response pipeline

```
[MimicRec POST /predict]
JSON body fields (per MimicRec contract YAML):
  image_primary: <base64 JPEG>     # already 224×224 RGB, no data: prefix
  image_wrist:   <base64 JPEG>     # optional per contract
  proprio:       [float, ...]      # raw, per-domain shape & units
                                   #   e.g., SO-101 = [j1..j5_deg, gripper_packed_deg, 0.0]
  instruction:   str               # UTF-8, may be empty in pre-start states
  model_version: "x_vla_v36"       # informational, server ignores for dispatch
  _t_mono_ns:    {state, image:*, instruction}   # optional staleness check
       ↓
[FastAPI handler]  inference_server.py:/predict
  - pydantic validate (PredictRequest)
  - reject malformed (4xx) — protects DomainAdapter from shape errors
       ↓
[DomainAdapter.preprocess]   driven by deploy yaml + ckpt meta
  - JPEG decode → np.ndarray[H, W, 3] uint8 RGB (one per camera key)
  - field-name mapping: image_primary → scene_image, image_wrist → wrist_image
  - per-domain proprio adapter:
      raw [D_robot] → model proprio [D_prop=8] f32
      (incl. units conversion deg→rad, padding/truncation, Q99 normalization
       using ckpt-baked stats)
  - returns Obs = {scene_image, wrist_image, wrist_was_provided: bool, proprio, language}
    where wrist_was_provided is True when MimicRec sent the wrist field, False when zero-filled
    for a dropout-enabled ckpt (per Section 4 zero-fill rule).
       ↓
[ChunkPredictor.predict]
  HoldPositionChunkPredictor:
    - emit zero ee_delta for cols 0..5, deploy.holdposition.gripper_native_midpoint
      for col 6 (default 0.5). The midpoint must be expressed in MODEL NATIVE
      gripper units; for v36 (RLDS native = normalized_0_1 with closed=0/open=1),
      0.5 lands on the contract midpoint after postprocess. Zero in last column
      would map to closed for that convention — not safe.
    - HoldPosition is a wire-shape smoke / pre-model-trained sentinel,
      NOT a production safety fallback (MimicRec's slow-stop is the real
      fallback). See Section 5 for the full implementation.
  XVLAAdapterChunkPredictor:
    - SigLIP transform per camera → torch[1, 3, 224, 224] bf16
    - tokenize language → input_ids, attention_mask
    - assemble internal batch dict (matches existing model.forward signature):
        {domain_id, scene_image, wrist_image, proprio, prompt_*,
         last_action_chunk: zeros[T, A], target_action: zeros, action_mask: ones,
         wrist_mask: obs["wrist_was_provided"]}     # carried from DomainAdapter.preprocess
    - DINOv2 conditional: include batch["wrist_image_dinov2"] /
      batch["scene_image_dinov2"] when model.cfg flags require (see Section 5)
    - with torch.no_grad(): pred, _ = model(batch)
    - apply Q99 action denormalization → physical units
  returns np.ndarray[T, A] f32 (native units/frame, e.g., for v36: LIBERO action format)
       ↓
[DomainAdapter.postprocess]
  - frame conversion if model_native_frame ≠ contract.frame
      (e.g., world → ee_local: chain ΔT through current ee pose if needed;
       MVP assumption: train models in the contract's frame to avoid runtime conversion)
  - gripper convention conversion (driven by deploy.action.{native,contract}.gripper):
      model native, post-Q99-denorm (e.g., RLDS normalized_0_1 with closed=0, open=1
      for v33/v35/v36 RLDS-trained) → contract.gripper.kind/units (e.g., SO-101 normalized_0_1 with
      closed=0, open=1 → identity). For sign-mismatched conventions
      (e.g., signed → normalized): g_out = (g_native - native.closed) / (native.open - native.closed)
      then re-scale to contract.range.
  - row-shape assert: each row exactly 7 floats (server-side guard against the
    "missing gripper column" client gap #4 in MimicRec doc)
       ↓
[FastAPI response]
  PredictResponse JSON:
    actions: [[Δx0, Δy0, Δz0, Δrx0, Δry0, Δrz0, g0], ...]   # length = T (= ckpt's action_chunk_len)
    # `done` field intentionally omitted (MimicRec gap #2: not consumed yet)
```

### Cross-cutting design decisions

- **Stateless inter-request:** `last_action_chunk` is fed as **zeros every call**. Rationale: HTTP requests carry no episode boundary signal, so maintaining state across calls would risk poisoning a new episode with a previous one's tail. Trade-off: loses the autoregressive feedback the model was trained with — acceptable for Phase 0; revisit if quality drops materially.
- **Image already 224×224, but full SigLIP transform still applies:** MimicRec resizes client-side to 224. The server feeds those 224×224 RGB tensors directly into `SiglipImageTransform`, which **internally** runs `Resize(248) + CenterCrop(224) + Normalize` — the same zoom-then-crop the training pipeline applies. Skipping the SigLIP transform's Resize/CenterCrop would create a deployment-time distribution mismatch against the v33/v35/v36 training distribution. The LIBERO-specific 256→224 pre-resize in `xvla_adapter_policy.py:_np_image_to_chw` is a no-op when the input is already 224 and is **kept** as a safety net against off-spec inputs.
- **Raw proprio handling:** MimicRec's documented gap #1 is that the client never normalizes proprio. Server-side normalization (Q99 from ckpt meta) is mandatory in `DomainAdapter.preprocess`.
- **Chunk size = ckpt's action_chunk_len:** v36 declares `cfg.data.action_chunk_len=8` in its train yaml; the resulting ckpt will emit chunks of length 8. The MimicRec contract YAML for this server **must** declare `chunk.expected_size: 8` (the user controls this YAML). Server returns whatever the model emits; MimicRec gap #3 (size-mismatch reject not enforced) means relying on the server to produce the right size.
- **HoldPosition emits zero ee_delta + a configured midpoint gripper in NATIVE units:** ee_delta zeros stay zero through postprocess (unit-invariant). The gripper column carries `deploy.holdposition.gripper_native_midpoint` (default 0.5) which is chosen so that postprocess produces the contract midpoint. Zero across all columns would silently command CLOSED for any `normalized_0_1` native convention (closed=0). HoldPosition does not derive a command from raw proprio. It is intentionally inert (not a safety fallback) — MimicRec's slow-stop ramp is the real fallback.
- **Failure isolation:**
  - Pydantic validation error → 422 (FastAPI default) — MimicRec logs `inference_error`, slow-stops.
  - Predictor raises (CUDA OOM, NaN) → 500, log full traceback, slow-stop on client.
  - DomainAdapter mapping error (e.g., wrong proprio length) → 422 with explicit message.

### Latency budget breakdown (target 266 ms)

Rough per-stage budget for v36 on RTX 5070 Ti 16 GB:

| Stage | Target |
|---|---|
| HTTP + pydantic decode | <5 ms |
| JPEG decode × 2 + numpy conversion | <10 ms |
| DomainAdapter.preprocess (proprio normalize, mapping) | <2 ms |
| SigLIP image transform × 2 + tokenize | <10 ms |
| Model forward (bf16, batch=1, T=8) | 200–230 ms |
| Action denormalize + DomainAdapter.postprocess | <5 ms |
| JSON encode + HTTP send | <5 ms |
| **Total** | **~250 ms** (slightly under target) |

If model forward exceeds 230 ms after `torch.compile` warmup, the headroom is gone — fallbacks would be reducing `H_act` (training-side change), pruning attention layers, or accepting slow-stop on a fraction of calls.

---

## Section 4. Deploy YAML schema + ckpt metadata contract

### Existing ckpt export format (no schema changes)

`~/X-VLA-Adapter_export/<run>/` already provides everything the server needs:

| File | Contents | Server use |
|---|---|---|
| `meta.json` | `{step, cfg, norm_stats, tokenizer_settings, git_commit}` | model arch + de/normalization stats (canonical source) |
| `dataset_statistics.json` | mirror of `meta.json["norm_stats"]` | **informational only**; server reads `meta.json` |
| `train_config.yaml` | YAML form of `meta.json.cfg` | reference / human read |
| `model.pt` | torch state_dict | weights |
| `eval_configs/` | pre-built eval YAMLs | n/a (eval-only) |

Server reads `meta.json` exclusively. No new training-side schema work — Phase 0 reuses the existing v33/v35 export pipeline; v36 will produce the same shape once trained.

> Verified against the existing v33/v35 export format (e.g., `v33_step40000/meta.json`): top-level keys are `step, cfg, norm_stats, tokenizer_settings, git_commit`. The norm_stats payload is shaped identically to `dataset_statistics.json` (`{<unnorm_key>: {action: {mean,std,q01,q99,mask}, proprio: {...}}}`) and is what `training/checkpoint.save_checkpoint` writes. v36 will use the same export pipeline.

### Deploy YAML schema (new file shape)

`configs/deploy/<robot>_<model>.yaml`. One file per (robot, ckpt) combination. Below is the full annotated template; required unless tagged `[opt]`.

```yaml
# --- 1. Identity / sanity assertions ---
ckpt:
  expected_unnorm_key: libero_spatial_no_noops       # must exist in meta.json["norm_stats"]; v36 trains on libero_spatial_no_noops only
  expected_action_chunk_len: 8                        # must equal cfg.data.action_chunk_len
  expected_action_dim: 7                              # MVP fixed: 6 ee_delta + 1 gripper
  expected_proprio_dim: 8                             # may differ from raw MimicRec proprio dim — proprio.adapt handles the gap

# --- 2. MimicRec contract field-name mapping ---
request_fields:
  scene_image: image_primary
  # wrist_image: REQUIRED unless ckpt is dropout-tolerant. Specifically:
  #   - Hard required (must be set): cfg.model.use_wrist_bridge=True OR
  #     getattr(cfg.model, "use_scene_wrist_dinov2_llm", False)=True OR
  #     any DINOv2/wrist-bridge flag (these paths read wrist outside the LLM
  #     slot, dropout does NOT mask them). v33/v35 land here.
  #   - Soft required (may be omitted): cfg.model.wrist_in_llm=True with
  #     cfg.model.wrist_view_dropout_p > 0 — only the LLM wrist slot consumes
  #     it, which dropout zero-masks at train time. v36 lands here.
  wrist_image: image_wrist
  proprio: proprio                                    # [opt, default = proprio]
  instruction: instruction                            # [opt, default = instruction]
  # Zero-fill behaviour at request time (only for soft-required ckpts like v36):
  #   - wrist_image absent in request → server fills np.zeros((224,224,3), uint8)
  #     and sets wrist_was_provided=False (which becomes wrist_mask=False).
  # For hard-required ckpts (v33/v35): startup hard-fails if request_fields.wrist_image
  # is missing; runtime returns 422 if a request omits the field.

# --- 3. Proprio adapter (raw MimicRec proprio → model proprio) ---
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
    method: q99                                       # none | q99
    stats_key: proprio                                # → dataset_statistics[unnorm_key].proprio

# --- 4. Action output (model native → MimicRec contract) ---
action:
  native:
    units: meter_axisangle_rad
    frame: world                                      # LIBERO RLDS-trained = world
    gripper:
      kind: absolute
      units: normalized_0_1                           # RLDS convention (verified vs xvla_adapter_policy.py:_refill_buffer)
      sign: { closed: 0, open: 1 }
  contract:
    units: meter_axisangle_rad
    frame: ee_local
    gripper:
      kind: absolute
      units: normalized_0_1
      sign: { closed: 0, open: 1 }
  denormalization:
    method: q99                                       # none | q99 | mean_std
    stats_key: action                                 # mask=False dims are passed through unchanged
  frame_conversion:
    method: none                                      # none | world_to_ee_local | ee_local_to_world
    # Required to be a real conversion when native.frame != contract.frame.
    # See `wire_only_smoke` below for the smoke-test escape hatch.

# --- 5. HoldPosition fallback (Phase 0 / pre-trained sentinel) ---
holdposition:
  gripper_native_midpoint: 0.5    # in MODEL NATIVE gripper units. 0.5 maps to
                                  # contract midpoint for normalized_0_1.
                                  # Use 0.0 for signed_neg1_pos1 native, etc.

# --- 5b. Smoke-test escape hatch (required for v36 → SO-101 wiring) ---
wire_only_smoke: false                                 # set true ONLY when motion is known-bad
# When true, server skips the native/contract frame and gripper-convention
# match assertions. Use ONLY for wire-shape smoke tests where you have
# already accepted that the model's actions will not move the robot
# correctly (e.g., LIBERO-Franka ckpt v36 against SO-101 contract).
# Default false → hard-fail at startup if (a) native.frame != contract.frame
# and frame_conversion.method == none, or (b) native.gripper.{units,sign}
# != contract.gripper.{units,sign} with no documented conversion.

# --- 6. Runtime knobs ---
runtime:
  device: cuda:0
  dtype: bf16
  torch_compile: off                                  # off | reduce-overhead | default
  warmup_iters: 1
```

### Startup validation flow

`inference_server.build_app(predictor_kind, checkpoint, deploy_config_path, domain_id, inject_sleep_s)` runs:

1. Load deploy_config (`yaml.safe_load` + pydantic model for schema validation).
2. If `predictor_kind == "xvla_adapter"`: load ckpt `meta.json` and run **all** hard-fail asserts in step 3. If `predictor_kind == "hold_position"`: skip ckpt load entirely; run only the deploy-yaml-internal subset of step 3 (those that don't reference `cfg.*` or `meta.norm_stats`):
   - `domain_id >= 0` (lower bound only — without ckpt we can't verify the upper bound; documented as a known limitation since HoldPosition's output is independent of `domain_id`)
   - `deploy.proprio.source.total_dim == sum(c.dims for c in source.components)` (deploy-yaml internal consistency)
   - `deploy.expected_action_chunk_len > 0` and `deploy.expected_action_dim == len(deploy.action.contract.gripper.kind == "delta" ? ... : 7)` — practically: `deploy.expected_action_dim == 7` for all MVP contracts.
   - Skip the wrist requirement check (HoldPosition does not read wrist).
   - Skip the frame mismatch check (HoldPosition's ee_delta is zero and frame-invariant).
   - Still apply gripper-convention compatibility check (HoldPosition gripper goes through `DomainAdapter.postprocess`); `wire_only_smoke=true` still bypasses it.
3. Hard-fail asserts (raise at startup, never serve a request):
   - `0 <= domain_id < cfg.model.num_domains` (lower bound matters; argparse type=int allows `-1`, which would silently index Python's last embedding row)
   - `cfg.data.unnorm_key == deploy.ckpt.expected_unnorm_key`
   - `resolved_action_chunk_len == deploy.ckpt.expected_action_chunk_len` where `resolved_action_chunk_len = cfg.model.get("action_chunk_len") or cfg.data.get("action_chunk_len") or C.ACTION_CHUNK_LEN`. v36 sets `cfg.data.action_chunk_len=8` explicitly (verified against `configs/train/libero_spatial_v36.yaml`); the fallback chain remains relevant for older ckpts (e.g., v35) where neither cfg block sets it.
   - `len(meta.norm_stats[unnorm_key].action.mean) == deploy.ckpt.expected_action_dim`
   - `len(meta.norm_stats[unnorm_key].proprio.mean) == deploy.ckpt.expected_proprio_dim`
   - `deploy.proprio.adapt.output_dim == deploy.ckpt.expected_proprio_dim`
   - `deploy.proprio.source.total_dim == sum(c.dims for c in source.components)`
   - **Wrist requirement** — split by which path consumes the wrist tensor:
     - **Hard required (dropout-irrelevant)**: `cfg.model.use_wrist_bridge` OR `getattr(cfg.model, "use_scene_wrist_dinov2_llm", False)` OR any other flag making `VLAPolicy.forward()` read wrist outside the LLM slot. These paths run unconditionally per forward; `wrist_view_dropout_p` does not gate them. → `deploy.request_fields.wrist_image` MUST be set; zero-fill at request time is rejected (would feed garbage features to the bridge).
     - **Soft required (dropout-tolerant)**: `cfg.model.wrist_in_llm=True` with `cfg.model.wrist_view_dropout_p > 0`. Dropout-trained — missing wrist is fine. → `deploy.request_fields.wrist_image` MAY be omitted; zero-fill + `wrist_was_provided=False` flows through.
     - **Other** (no wrist consumption): no constraint.
4. Frame & gripper compatibility (hard-fail unless `wire_only_smoke: true`):
   - If `native.frame != contract.frame` and `frame_conversion.method == none` → fail. Either train in contract frame, set `frame_conversion.method` to an implemented converter, or set `wire_only_smoke: true` if accepting wrong motion for smoke.
   - If `native.gripper.{units, sign}` does not match `contract.gripper.{units, sign}` and the deploy yaml has no explicit conversion description (currently only `signed_neg1_pos1 ↔ normalized_0_1` is implemented as a built-in, plus invert if `sign.open` differs) → fail. Same `wire_only_smoke` escape applies.
5. Soft-warn (log + continue):
   - Default deploy yaml (e.g., a copy of `_template.yaml`) without operator-specific edits → log a one-line warning that the file looks unmodified.
6. Construct `DomainAdapter`:
   - `xvla_adapter`: `DomainAdapter(deploy_config, meta.norm_stats[unnorm_key], domain_id)`.
   - `hold_position`: `DomainAdapter(deploy_config, norm_stats=None, domain_id)` — proprio Q99 normalization is skipped (HoldPosition's batch is unused for predict; only postprocess gripper conversion + row-shape assert run).
7. Construct `ChunkPredictor`:
   - `hold_position`: `HoldPositionChunkPredictor(chunk_len=deploy.expected_action_chunk_len, action_dim=deploy.expected_action_dim, gripper_native_midpoint=deploy.holdposition.gripper_native_midpoint)`.
   - `xvla_adapter`: `XVLAAdapterChunkPredictor(runtime=ModelRuntime.from_export(checkpoint, ...), tokenizer, image_transform, action_q99=meta.norm_stats[unnorm_key].action, action_chunk_len, action_dim, domain_id)`.
8. Warmup × `runtime.warmup_iters` — `xvla_adapter` only; `hold_position` is a no-op (zeros + scalar; no torch path).
9. Mount FastAPI route → return app.

### Split rationale

- **Numeric stats in ckpt** because they are tied to the training run; mismatched stats silently corrupt actions.
- **Field names / gripper convention / frame in deploy yaml** because they describe the *deployment side* (which MimicRec contract this server is paired with) and have no causal link to training. The same v36 ckpt could serve different real-robot setups via different deploy yamls.
- **Both sides validated against each other at startup** (`expected_*` fields) so a wrong pairing fails before the first request, not silently mid-rollout.

> **Cross-reference:** the `holdposition.gripper_native_midpoint` field (Section 4 template, default 0.5) feeds `HoldPositionChunkPredictor` so the gripper column lands on the contract midpoint rather than zero (which would command closed for any `normalized_0_1` native convention).

---

## Section 5. ChunkPredictor & ModelRuntime

### `ChunkPredictor` ABC

```python
# src/vla_project/deployment/predictors/base.py
from abc import ABC, abstractmethod
import numpy as np

class ChunkPredictor(ABC):
    """Returns one action chunk in MODEL NATIVE physical units.

    Input: Obs dict already passed through DomainAdapter.preprocess —
      scene_image: np.uint8[H, W, 3]
      wrist_image: np.uint8[H, W, 3] | None
      proprio:     np.float32[D_prop]      (already adapted + Q99-normalized)
      language:    str

    Output: np.float32[T, A] in model native units. For v36 (and v33/v35
    RLDS-trained): gripper is normalized_0_1 (closed=0, open=1), frame is
    LIBERO world frame, deltas are meter+axisangle_rad. Frame / gripper-
    convention conversion happens AFTER this call in DomainAdapter.postprocess.
    """
    @abstractmethod
    def predict(self, obs: dict) -> np.ndarray: ...

    @property
    @abstractmethod
    def chunk_len(self) -> int: ...

    @property
    @abstractmethod
    def action_dim(self) -> int: ...
```

### `HoldPositionChunkPredictor`

```python
class HoldPositionChunkPredictor(ChunkPredictor):
    def __init__(
        self,
        chunk_len: int,
        action_dim: int,
        gripper_native_midpoint: float = 0.5,   # from deploy.holdposition.gripper_native_midpoint
    ):
        self._T = chunk_len
        self._A = action_dim
        self._g = float(gripper_native_midpoint)

    def predict(self, obs: dict) -> np.ndarray:
        a = np.zeros((self._T, self._A), dtype=np.float32)
        a[:, -1] = self._g                       # native-units midpoint in last column
        return a
```

After postprocess:

- ee_delta zeros → zeros (unit-invariant; no movement)
- gripper `_g` (native midpoint) → contract midpoint via the deploy yaml's gripper conversion (linear maps preserve midpoint)

Defaults: `gripper_native_midpoint = 0.5` is correct for `normalized_0_1` native (closed=0, open=1) — used by every RLDS-trained ckpt including v33/v35/v36. For `signed_neg1_pos1` native, set `0.0` in the deploy yaml.

Caveat: HoldPosition is for wire-shape smoke only. The "midpoint" is "ambiguous-but-not-fully-actuating" — for `binary_threshold_0p5` contracts the postprocess thresholds 0.5 to either 0 or 1 depending on the threshold tie-break. Pick a midpoint that lands on the safer side or do not use HoldPosition for that contract.

### `XVLAAdapterChunkPredictor`

```python
class XVLAAdapterChunkPredictor(ChunkPredictor):
    def __init__(
        self,
        runtime: ModelRuntime,
        tokenizer: GemmaPromptTokenizer,
        image_transform: SiglipImageTransform,
        action_q99: Q99Stats,                 # from ckpt dataset_statistics.json
        action_chunk_len: int,
        action_dim: int,
        domain_id: int,
    ): ...
```

`predict(obs)` steps:

1. SigLIP transform per camera → bf16 `[1, 3, 224, 224]` (full `Resize(248) + CenterCrop(224) + Normalize`).
2. Tokenize `obs["language"]` → `input_ids`, `attention_mask`.
3. `proprio` already normalized → `bf16[1, D_prop]`.
4. Assemble batch (matches existing `model.forward` signature):
   ```
   { domain_id: long[1],
     scene_image, wrist_image, proprio,
     prompt_input_ids, prompt_attention_mask,
     last_action_chunk: zeros[1, T, A],     # stateless — see Section 3
     target_action: zeros, action_mask: ones,
     wrist_mask: obs["wrist_was_provided"],  # carried from DomainAdapter.preprocess; False when zero-filled for dropout-enabled ckpt
   }
   ```
   **Conditional DINOv2 inputs** (mirrors existing `XVLAAdapterPolicy._build_batch`):
   - If `getattr(model, "wrist_dinov2_encoder", None) is not None`:
     `batch["wrist_image_dinov2"] = DINOv2ImageTransform(wrist_image)` (bf16, [1, 3, 224, 224])
   - If `getattr(model.cfg, "use_scene_wrist_dinov2_llm", False)`:
     `batch["scene_image_dinov2"] = DINOv2ImageTransform(scene_image)` (bf16, [1, 3, 224, 224])

   v33/v35/v36 do not enable either flag — these branches are no-ops. Required for v32 and future DINOv2-enabled ckpts.
5. `with torch.no_grad(): pred, _ = runtime(batch)` → `pred [1, T, A] bf16`.
6. `.cpu().float().squeeze(0)` → apply Q99 denormalization (mask-aware: dims with `mask=False` pass through) → `np.float32[T, A]` in model-native physical units.
7. Return.

**No** gripper sign flip / scale here — that lives in `DomainAdapter.postprocess`.

### `ModelRuntime`

```python
# src/vla_project/deployment/runtime.py
class ModelRuntime:
    @classmethod
    def from_export(
        cls,
        ckpt_dir: Path,
        device: str = "cuda:0",
        dtype: torch.dtype = torch.bfloat16,
        torch_compile: Literal["off", "reduce-overhead", "default"] = "off",
        warmup_iters: int = 1,
    ) -> "ModelRuntime": ...

    def __call__(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor]: ...   # (pred, loss)
```

Responsibilities:

- Build `VLAPolicy` from `meta.json.cfg` (existing `training/checkpoint.load_checkpoint`).
- Move to `device`, cast to `dtype`, set `eval()`.
- Optionally wrap in `torch.compile(mode=torch_compile, fullgraph=False)` (first call cost 30–60 s, amortized across requests).
- Warmup: synthesize a zero-image / zero-proprio / padded-prompt batch and run `forward` × `warmup_iters` to JIT-stabilize and pre-allocate KV cache buffers.
- `__call__` is the single forward used by predictors; raises a custom `InferenceError` on torch failure so the FastAPI handler returns 500 cleanly.

### Reuse with existing `XVLAAdapterPolicy`

`policies/xvla_adapter_policy.py` and `deployment/predictors/xvla_adapter.py` initially inline the shared SigLIP-apply / tokenize / batch-dict-build code separately. Extraction into a shared helper happens only when duplication causes drift — YAGNI per CLAUDE.md "Coding Rules".

---

## Section 6. Failure handling, observability, testing, Phase 0 acceptance

### Failure handling

Per-stage failure → HTTP code mapping. Server never silently drops; every failed request produces a 4xx/5xx so MimicRec sees the error and triggers slow-stop.

| Failure | HTTP | MimicRec | Server log level |
|---|---|---|---|
| Pydantic validation (malformed JSON, wrong types, missing required field) | 422 | `inference_error` event → slow-stop next tick | warn (1 line / request) |
| DomainAdapter.preprocess (proprio length mismatch, JPEG decode error, `deg_to_rad` on non-numeric) | 422 | same | warn with field name |
| `wrist_image` absent for hard-required ckpt (use_wrist_bridge / DINOv2 paths) | 422 | same | warn with ckpt path |
| ChunkPredictor.predict raises (CUDA OOM, NaN sentinel) | 500 | same | error + traceback |
| DomainAdapter.postprocess (gripper convention out of declared range, `frame_conversion.method` not implemented) | 500 | same | error + traceback |
| Action row width assert (server-internal contract guard, should never fire) | 500 | same | error + traceback |
| Latency exceeds 0.5 × chunk_duration (target 266 ms for chunk_size=8 @ 15 fps) | still 200 — but emit warning log line + per-stage timings | MimicRec's 5 s `client_timeout` (per spec) covers catastrophic stalls; soft over-budget is observed via `latency_budget_exceeded: true` in the server's own log + p95 in the latency benchmark | warn with `elapsed_ms`, `latency_budget_ms`, `latency_budget_exceeded: true` |

Notes:
- 4xx = client/contract error (request malformed or incompatible with deploy config). 5xx = server bug or transient torch failure.
- Server does not retry internally. MimicRec also does not retry.
- A single NaN in the predicted chunk → 500 (refuse to emit), not silent zero-fill.
- "Server still returns 200 on slow responses" relies on (a) MimicRec's documented 5 s `client_timeout` cutting off truly stuck requests, and (b) MimicRec's slow-stop ramping when the chunk arrives too late to feed the dispatcher. Both are MimicRec-side concerns, **not server-side**. The Phase 0 server-side test verifies only that the server emits the `latency_budget_exceeded: true` log line and still returns 200 within 5 s; whether MimicRec actually triggers slow-stop is verified separately during MimicRec rollout testing (out of scope for this server's acceptance gate).

### Observability

**Per-request structured JSON log** (one line):

```json
{
  "ts_ns": 488772692451200,
  "request_id": "01JE...",
  "elapsed_ms": 247.3,
  "predictor": "XVLAAdapterChunkPredictor",
  "domain_id": 0,
  "outcome": "ok"
}
```

`outcome` ∈ `{"ok", "invalid_request", "predictor_error", "postprocess_error"}`. On failure, append `error_class` + `error_msg`; full traceback stays at error-level only (off by default in production).

**Startup log block** (one-shot). Fields scoped by predictor:

Always logged:
- `predictor_class, deploy_config_path, domain_id, host, port`
- `wrist_requirement (hard|soft|none)`  (resolved from deploy yaml + ckpt cfg if present, else from deploy yaml only)
- `resolved_action_chunk_len, action_dim, proprio_dim`

`xvla_adapter` only:
- `ckpt_dir`
- `meta.json: step, git_commit`
- `norm_stats: unnorm_key, action_dim, proprio_dim`
- `runtime: device, dtype, torch_compile, warmup_iters`

`hold_position` only:
- `gripper_native_midpoint, inject_sleep_s` (the test-only flag value, default 0)

**Endpoints**:

- `POST /predict`: the contract route.
- `GET /healthz`: `{"status": "ok|warming|error", "predictor": "...", "ready_at_ns": ...}`. Used by orchestration to gate traffic; returns `warming` until startup warmup × `runtime.warmup_iters` completes.

**What NOT to log**:
- Image bytes (privacy + size).
- Raw proprio values (privacy if real-robot data).
- Instruction text (only at `--debug` level — task descriptions can be sensitive).

### Testing

Phase 0 test pyramid (CLAUDE.md "Tests" minimum + deployment-specific):

| Path | Must explicitly cover |
|---|---|
| `tests/test_deployment_schemas.py` | pydantic round-trip; missing required field → 422; type coercion (str→base64 bytes); extra fields tolerated |
| `tests/test_domain_adapter.py` | proprio.adapt ops (`deg_to_rad`, `copy`, `pad_zeros`); Q99 normalize with mask (mask=False dim passes through); gripper convention conversion (`normalized_0_1` identity; `signed_neg1_pos1` ↔ `normalized_0_1`; sign-flip when `sign.open` differs); `frame_conversion=none` identity; **postprocess row-width assert: input `[T, 7]` ok, `[T, 6]` raises, `[T, 8]` raises** |
| `tests/test_predictor_holdposition.py` | output shape exactly `(T=8, A=7)`; cols 0..5 == 0; col 6 == `gripper_native_midpoint` (default 0.5); reset() is no-op |
| `tests/test_predictor_xvla_adapter.py` | (fake `ModelRuntime` returning deterministic tensor) batch dict assembly; DINOv2 conditional keys present iff `model.cfg.use_scene_wrist_dinov2_llm`/`wrist_dinov2_encoder` set; `wrist_was_provided=False` → `wrist_mask=False`; Q99 denorm respects mask |
| `tests/test_runtime_load.py` | meta.json parsing; norm_stats extraction; **startup hard-fail assertions: `domain_id == -1`, `domain_id == cfg.model.num_domains`, `cfg.data.unnorm_key != deploy.expected_unnorm_key`, `resolved_action_chunk_len != deploy.expected_action_chunk_len`, `len(norm_stats[k].action.mean) != deploy.expected_action_dim`, `len(norm_stats[k].proprio.mean) != deploy.expected_proprio_dim`, `deploy.proprio.adapt.output_dim != deploy.expected_proprio_dim`, hard-required wrist mapping missing, frame mismatch without `wire_only_smoke=True`** |
| `tests/test_serve_smoke.py` | FastAPI `TestClient` end-to-end via HoldPosition: (a) valid request → 200 with `len(actions) == 8` and `len(actions[0]) == 7` and `actions[0][:6] == [0, 0, 0, 0, 0, 0]`; (b) missing required `scene_image` → 422; (c) `proprio` length wrong → 422; (d) hard-required `wrist_image` field declared in deploy yaml but absent in request → 422; (e) injected slow predictor (sleep) → 200 with `latency_budget_exceeded: true` in log |

**Integration smoke** (manual / CI gate, GPU optional). Phase 0 runs ONLY the HoldPosition path:

- Start server with `--predictor hold_position` (explicit flag; not derived from ckpt absence) so the gate is unambiguous: `python scripts/serve.py --predictor hold_position --deploy-config configs/deploy/v36_libero_spatial.yaml --domain-id 0 --port 8001`. (No `--checkpoint` required in this mode.)
- Run MimicRec's `scripts/smoke_inference_real_data.py` against `http://127.0.0.1:8001/predict`.
- Expected (HoldPosition only): all of:
  - `✅ inference mock pipeline works end-to-end with real data`
  - `IK failures: 0/N` (zero ee_delta → no IK displacement)
  - response payload `actions` has shape `[8, 7]` for every call
  - `actions[i][:6] == [0, 0, 0, 0, 0, 0]` for all `i`
  - `actions[i][6] ≈ contract midpoint` (i.e., what `holdposition.gripper_native_midpoint` becomes after `DomainAdapter.postprocess`; for v36 native + SO-101 contract both `normalized_0_1`, midpoint = 0.5 unchanged)
- v36 → SO-101 cross-frame run is **not a Phase 0 gate** — that is `wire_only_smoke` territory, with expected non-zero IK failures, deferred to Phase 1.

**Live latency benchmark** (post-v36 train, separate gate):

- `tools/benchmark_inference.py` synthesizes 1000 sequential requests, measures p50/p95/p99, reports per-stage breakdown.
- Acceptance: p95 < 266 ms after warmup, ≥99% success on synthetic input.

### Phase 0 acceptance gate

Phase 0 is "green" when ALL of:

1. Named deployment test files all pass: `uv run pytest tests/test_deployment_schemas.py tests/test_domain_adapter.py tests/test_predictor_holdposition.py tests/test_predictor_xvla_adapter.py tests/test_runtime_load.py tests/test_serve_smoke.py -q`. (Avoid `tests/test_*.py` glob — it tolerates accidentally missing files.)
2. `uv run python scripts/serve.py --predictor hold_position --deploy-config configs/deploy/v36_libero_spatial.yaml --domain-id 0 --port 8001` starts and `GET /healthz` returns `{"status": "ok"}` within 30 s.
3. MimicRec `smoke_inference_real_data.py` against the running server prints the success string AND the run reports the four extra HoldPosition assertions from the integration smoke (response shape `[8, 7]`, `actions[i][:6] == 0`, `actions[i][6] ≈ midpoint`, `IK failures: 0/N`).
4. Slow-response synthetic test: with `--predictor hold_position --inject-sleep 0.4` (seconds, test-only flag), one request returns 200 within MimicRec's 5 s timeout AND the server log emits `latency_budget_exceeded: true`. Verifies server-side observability only — MimicRec slow-stop behavior is verified separately in MimicRec rollout testing, not this gate.
5. `configs/deploy/_template.yaml` exists and is human-reviewed.
6. `src/vla_project/deployment/README.md` documents:
   - Server startup command (HoldPosition vs XVLAAdapter mode).
   - How to swap ckpts (kill + restart; no reload endpoint).
   - How to add a new robot deploy yaml.
   - Known limitations: frame conversion not implemented (use `wire_only_smoke: true` for cross-frame smoke); HoldPosition is not a safety fallback.

**Not in Phase 0** (deferred):
- Live v36 ckpt forward path (depends on v36 training completing).
- Latency benchmark.
- `world ↔ ee_local` frame converter implementation.
- `/admin/reload`, hot-swap, `max_inflight > 1`.
- DINOv2 / multi-domain ckpt explicit smoke.
