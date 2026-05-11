# X-VLA-Adapter

X-VLA / VLA-Adapter style Vision-Language-Action policy on top of
**SigLIP + Gemma4-E2B + per-domain projectors + L1 action head**, targeting
LIBERO benchmarks and (eventually) real-robot deployment.

## Results — `LIBERO-Spatial` (50 ep / suite, headless MuJoCo)

| version | architecture                                               | best step | peak SR | step_10000 |
|---------|------------------------------------------------------------|-----------|---------|------------|
| baseline (vla-gemma-4) | ResNet-18 wrist + Gemma4 + L1 head            | 10000     | 42 %    | 42 % |
| v25     | Mode B + image preproc fix + warmup gripper                | 10000     | 74 %    | 74 % |
| v28     | LoRA r=16 + AQ trainable + two-step warmup                 | 7500      | 64 %    | 60 % |
| v30     | v28 + tweaks                                               | 7500      | 70 %    | 32 % |
| v31     | SigLIP + DINOv2 scene + wrist-into-LLM (no LoRA)           | 10000     | 36 %    | 36 % |
| v32     | v31 + LoRA + AQ trainable (max_steps=20000)                | 17500     | 76 %    | 72 % |
| **v33** | **DA-2-MLP + soft-prompt-in-LLM + LoRA r=64 (40000 steps)**| **40000** | **94 %**| 24 % |

v33 trajectory: 5k → 16% / 10k → 24% / 15k → 66% / 20k → 74% / 25k → 62% /
30k → 80% / 35k → 84% / **40k → 94 %** (47/50). Soft prompts + DA-MLP + bigger
LoRA need long training to ramp; pre-20k SR is misleading.

In flight (no eval yet):

- **v34 / v35** — 4-suite multi-domain RLDS (v35 = shared Q99 stats fix).
  v34 step_40000 spatial = 2 % so far → multi-domain still under-trained or
  needs different schedule.
- **v36** — v33 base + π₀-style wrist-into-LLM (fixed 256-tok slot + mask +
  view-dropout 0.3); wrist_bridge path dropped. Currently training on dl40 GPU 4.

## Setup

```bash
bash scripts/setup.sh
```

Installs uv (if missing), syncs deps via `uv sync --extra dev`, initialises
the `VLA-Adapter` and `X-VLA` submodules, and runs a torch + Gemma4 smoke
check. See [`scripts/setup.sh`](scripts/setup.sh) for details.

Pre-requisites the script does **not** handle (host-specific):

- a sibling `vla-gemma-4/` checkout providing RLDS data + baseline checkpoints
- LIBERO simulator + assets (uses `MUJOCO_GL=osmesa` for headless render)
- Hugging Face token (`uv run huggingface-cli login`) for Gemma4 / SigLIP

## Repo layout

See [`CLAUDE.md`](CLAUDE.md) for the canonical layout + coding rules.
TL;DR:

```
src/vla_project/
  data/        # dataset → internal batch schema (RLDS, LeRobot, LIBERO)
  models/      # vision, language, projectors, action heads, vla_policy
  policies/    # runtime obs → action wrappers (XVLAAdapterPolicy)
  training/    # trainer, optim, schedulers, checkpoint, distributed
  evaluation/  # libero_eval, rollout, metrics
  robots/      # base / sim / lerobot I/O
configs/
  train/       # libero_*_v{N}.yaml — one file per architecture revision
  eval/        # libero_*_v{N}_step{K}.yaml — one per ckpt × suite to evaluate
docs/architectures/  # mermaid + svg diagrams (v32, v35, ...)
```

## Training

Single-GPU (current default — multi-GPU NCCL is host-unstable):

```bash
CUDA_VISIBLE_DEVICES=0 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
uv run python scripts/train.py configs/train/libero_spatial_v33.yaml
```

Run names + checkpoints land under `outputs/<wandb.name>/`.

## Evaluation

```bash
uv run python scripts/eval.py configs/eval/libero_v33_step10000.yaml
```

Eval rolls out 50 episodes per suite; metrics + per-task results write to
`outputs/<run>/eval/`. MP4 videos are optional (`save_video: true`).

## Inference server

FastAPI HTTP server hosting an X-VLA-Adapter checkpoint behind MimicRec's
`POST /predict` contract. Two predictor modes:

- **`hold_position`** (Phase 0) — emits a constant action chunk. No GPU / no
  ckpt required. Use for wire-format smoke testing.
- **`xvla_adapter`** (Phase 1) — loads a real ckpt and runs forward
  passes; returns the model's denormalized action chunk in NATIVE units.
  Requires a checkpoint dir (`meta.json` + `model.pt`) + matching deploy
  YAML.

### HoldPosition smoke (no GPU)

```bash
uv run python scripts/serve.py \
  --predictor hold_position \
  --deploy-config configs/deploy/v36_libero_spatial.yaml \
  --domain-id 0 \
  --port 8001

curl http://127.0.0.1:8001/healthz
# {"status":"ok","predictor":"HoldPositionChunkPredictor","ready_at_ns":...}
```

### XVLAAdapter (real ckpt)

The deploy YAML pins the ckpt's expected dims + describes the
proprio/action adapter (raw client units → model units → MimicRec contract
units). See [`configs/deploy/_template.yaml`](configs/deploy/_template.yaml)
for the schema and [`configs/deploy/so101_v46.yaml`](configs/deploy/so101_v46.yaml)
for a working SO101 example.

`--checkpoint` accepts either a local directory or a Hugging Face repo
id (`org/repo` or `org/repo/subfolder`). When given an HF id,
`ModelRuntime.from_export` calls `huggingface_hub.snapshot_download` and
caches under `~/.cache/huggingface/hub/`; subsequent loads are free.

```bash
# Load directly from Hugging Face Hub (preferred for deploy: reproducible,
# no local-state coupling).
CUDA_VISIBLE_DEVICES=5 \
  uv run python scripts/serve.py \
    --predictor xvla_adapter \
    --checkpoint takaki99/so101-v46/step_2000 \
    --deploy-config configs/deploy/so101_v46.yaml \
    --domain-id 9 \
    --host 127.0.0.1 \
    --port 8001

# Or use a local checkpoint directory (faster for in-development iteration).
CUDA_VISIBLE_DEVICES=5 \
  uv run python scripts/serve.py \
    --predictor xvla_adapter \
    --checkpoint outputs/so101_v46_step30k_ft_dl50/checkpoints/step_2000 \
    --deploy-config configs/deploy/so101_v46.yaml \
    --domain-id 9 \
    --host 127.0.0.1 \
    --port 8001

curl http://127.0.0.1:8001/healthz
# {"status":"ok","predictor":"XVLAAdapterChunkPredictor","ready_at_ns":...}

# /predict body: PredictRequest schema = {image_primary, image_wrist (b64 JPEG),
#   proprio (raw client units, per deploy yaml proprio.source), instruction}
# Response: {"actions": list[list[float]]}  shape (T, A) in native units.
```

Available HF ckpts:
- [`takaki99/so101-v46/step_2000`](https://huggingface.co/takaki99/so101-v46/tree/main/step_2000) — early FT checkpoint (smoke-verified, not yet evaluated on real robot)

A minimal smoke client lives nowhere yet; the test harness in
`tests/deployment/` exercises both predictor paths and is the easiest
reference for building a request.

Per-request latency on a single RTX 6000 Ada with bf16 + `torch_compile: off`
is ~220 ms (budget 266 ms, logged as a warning if exceeded).

Known limitations:

- `XVLAAdapterChunkPredictor` currently feeds zeros for
  `batch["last_action_chunk"]` — the model ignores that field
  (`vla_policy.py:530-537`, `x_init=zeros` in the action head). Streaming
  history persistence is therefore inert; revisit if a future arch
  reinstates the LastAction projection.
- The frame conversion adapter (`action.frame_conversion.method`) only
  supports `none`; `world_to_ee_local` / `ee_local_to_world` are stubs.
  Mark `wire_only_smoke: true` in the deploy YAML when the native frame
  does not equal the contract frame (e.g. SO101 v46 native is dataset
  "world" but a real SO101 MimicRec contract may want `ee_local`).

For details (deploy yaml authoring, ckpt swap, known limitations) see
[`src/vla_project/deployment/README.md`](src/vla_project/deployment/README.md).
Design + plan live under [`docs/superpowers/`](docs/superpowers/).

### SO101 fine-tuning + deploy walkthrough

The full SO101 path (HF dataset → v2.1 conversion → norm stats → FT →
serve) is automated by these tools:

```bash
# 1. Convert HF v3.0 → v2.1 layout consumable by lerobot 0.3.3,
#    and drop episodes with success=False / deleted=True.
uv run python tools/convert_so101_v3_to_v21.py \
  --repo_id takaki99/test_so101 \
  --out_root data/converted/takaki99_test_so101_v21

# 2. Compute Q99 stats for EE-delta action (with SO(3) logmap for d_rotvec
#    — plain subtraction wraps around at ±π) + EE-pose proprio.
uv run python tools/compute_norm_stats_so101.py \
  --converted_root data/converted/takaki99_test_so101_v21 \
  --dataset_key so101_test \
  --output data/norm_stats/so101_test.json

# 3. FT from a pretrained ckpt (resume_da_row_init=random for the new
#    domain — never copy_row_<n> for a fresh embodiment, see CLAUDE.md
#    "DA Row Init for FT").
CUDA_VISIBLE_DEVICES=4 \
  uv run accelerate launch \
    --config_file configs/accelerate/dl50_1gpu.yaml \
    --main_process_port 29501 \
    scripts/train.py configs/train/so101_v46_step30k_ft_dl50.yaml

# 4. Serve the FT'd ckpt (see XVLAAdapter section above).
```

## Configuration model

Every architecture revision is a config file, not scattered code edits.

- **Train** — `configs/train/libero_<suite>_v<N>.yaml`
  fields: `model.*` (arch flags), `vision.*`, `language.*`, `data.*`,
  `train.*` (lr, freeze schedule, lr_coefs per param group), `wandb.*`.
- **Eval** — `configs/eval/libero_v<N>_step<K>.yaml` references a trained
  checkpoint by directory + step.

Architecture diffs over time live in [`docs/architectures/*.mmd`](docs/architectures).

## Known quirks

- **cu128 wheels** are pinned (driver ≥ 12.6 hosts; Blackwell sm_120). See
  `[tool.uv.sources]` in `pyproject.toml`.
- **transformers ≥ 5.0** required for Gemma4 `model_type` registration;
  lerobot's `<5.0` cap is bypassed via `[tool.uv] override-dependencies`.
- **ROS2 on `PYTHONPATH`** breaks pytest plugin discovery — prefix tests
  with `PYTHONPATH=""`. See [`DEVELOPMENT.md`](DEVELOPMENT.md).
- **NCCL multi-GPU** is currently broken on hosts with NVML driver/library
  mismatch (e.g. dl40 post-2026-05-04 apt upgrade). Train single-GPU.
- **Eval EGL** likewise depends on a working GL stack; if `dl40` is broken,
  fall back to `dl42` (NFS-mounted same paths).

## Development

```bash
PYTHONPATH="" uv run pytest -v        # tests
uv run ruff check src/ tests/         # lint
```

See [`DEVELOPMENT.md`](DEVELOPMENT.md) and [`CLAUDE.md`](CLAUDE.md).
