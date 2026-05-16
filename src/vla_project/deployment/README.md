# `vla_project.deployment` — VLA Inference HTTP Server

Hosts an X-VLA-Adapter checkpoint behind MimicRec's `POST /predict` contract.

The server returns **model-native, fully q99-denormalized action chunks** with optional per-checkpoint `post_process.py` applied. Contract translation (frame conversion, gripper convention mapping, raw proprio adaptation) is the client's responsibility.

**Design:** see [`docs/superpowers/specs/2026-05-16-yamlless-hf-deploy-design.md`](../../docs/superpowers/specs/2026-05-16-yamlless-hf-deploy-design.md) (supersedes the prior 2026-05-06 spec for deploy YAML structure and contract division).

---

## Starting the server

### HoldPosition mode (no GPU / no ckpt required)

```bash
uv run python scripts/serve.py \
  --predictor hold_position \
  --port 8001
```

Verify:
```bash
curl http://localhost:8001/healthz
# {"status":"ok","predictor":"HoldPositionChunkPredictor","ready_at_ns":...}
```

### XVLAAdapter mode (real inference)

```bash
uv run python scripts/serve.py \
  --checkpoint takaki99/GEM-4-FT-bottle \
  --trust-checkpoint-code \
  --port 8001

curl http://localhost:8001/healthz
# {"status":"ok","predictor":"XVLAAdapterChunkPredictor","ready_at_ns":...}

# Or use a local checkpoint directory:
uv run python scripts/serve.py \
  --checkpoint outputs/so101_v46_step30k_ft_dl50/checkpoints/step_2000 \
  --port 8001
```

GPU is required for `xvla_adapter` mode.

---

## Swapping checkpoints

There is no reload endpoint. Kill the process and restart with the new
`--checkpoint` value. systemd / docker restart policies handle the gap.

---

## Client contract division

The server returns **model-native units** (fully q99-denormalized). The client is
responsible for:

1. **Proprio adaptation** — map raw robot sensors to the model's expected proprioceptive format (e.g., [ee_x, ee_y, ee_z, rotvec_x, rotvec_y, rotvec_z, gripper, pad]).
2. **Frame conversion** — transform actions from model-native frame (e.g., world-absolute) to the target robot frame (e.g., ee-local).
3. **Gripper convention** — if the model predicts normalized gripper [0, 1] but the robot expects raw gripper pwm or voltage, apply the post_process script (auto-loaded if `--trust-checkpoint-code` is passed) or implement conversion in the client.

See the spec for detailed examples.

---

## Known limitations

- **`max_inflight=1` only** (matches MimicRec's MVP setting). No request batching.
- **Latency target:** p95 < 266 ms after `torch.compile` warmup on RTX 6000 Ada with bf16.

---

## Test suite

Run the yamlless deploy test suite:

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
