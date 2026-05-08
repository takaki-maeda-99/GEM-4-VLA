# `vla_project.deployment` — VLA Inference HTTP Server

Hosts an X-VLA-Adapter checkpoint behind MimicRec's `POST /predict` contract.

**Design:** see `docs/superpowers/specs/2026-05-06-vla-inference-server-design.md`.
**Implementation plan:** see `docs/superpowers/plans/2026-05-06-vla-inference-server-phase0.md`.

---

## Phase status

- **Phase 0 (this code):** Skeleton + `HoldPositionChunkPredictor` end-to-end. `XVLAAdapterChunkPredictor` is a stub raising `NotImplementedError("Phase 1")`. `ModelRuntime.__call__` likewise. No GPU required, no real ckpt required.
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

For the MimicRec integration smoke (Phase 0 acceptance gate item 3):

```bash
# Terminal 1: start the server
uv run python scripts/serve.py \
  --predictor hold_position \
  --deploy-config configs/deploy/v36_libero_spatial.yaml \
  --domain-id 0 --port 8001

# Terminal 2: copy the pairing example into MimicRec's contract dir, then run smoke
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
