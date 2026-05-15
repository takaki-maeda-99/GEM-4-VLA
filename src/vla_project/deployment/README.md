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

## Adding a new robot deployment

Robot-specific config (proprio source/adapt, action frame/gripper convention) is no longer loaded from YAML files. Instead, it is baked into the checkpoint's `meta.json` during training and passed directly to the runtime via CLI arguments or environment variables.

See `scripts/serve.py --help` and the spec `docs/superpowers/specs/2026-05-06-vla-inference-server-design.md` for details on providing robot-specific metadata at deployment time.

---

## Known limitations (Phase 0)

- **Frame conversion not implemented.** Set `wire_only_smoke: true` for cross-frame deploy yamls (motion will be wrong; useful only for wire-format smoke).
- **HoldPosition is not a safety fallback.** It is for wire-shape smoke / pre-model-trained sentinel. MimicRec's slow-stop ramp is the real fallback for missing-action conditions.
- **No `/admin/reload` endpoint.** Kill + restart for ckpt swap.
- **`max_inflight=1` only** (matches MimicRec's MVP setting). No request batching.
- **Latency benchmark deferred.** Target p95 < 266 ms after `torch.compile` warmup; not measured in Phase 0.

---

## Phase 0 acceptance verification

Run the core yamlless test suite:

```bash
PYTHONPATH= uv run pytest \
  tests/test_wire_io_denorm.py \
  tests/test_wire_io_jpeg.py \
  tests/test_wire_io_proprio_ood.py \
  tests/test_post_process_loader.py \
  tests/test_startup_validation.py \
  tests/test_metadata_response.py \
  tests/test_runtime_post_process.py \
  tests/test_runtime_load.py \
  tests/test_inference_server_yamlless.py \
  tests/test_checkpoint_native_action.py \
  tests/test_backfill_native_action.py \
  -v
```

Expected: 50 passed.
