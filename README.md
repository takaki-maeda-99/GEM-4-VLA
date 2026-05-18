# X-VLA-Adapter

X-VLA / VLA-Adapter style Vision-Language-Action policy on top of
**SigLIP + Gemma4-E2B + per-domain projectors + L1 action head**, targeting
LIBERO benchmarks and (eventually) real-robot deployment.

## Results — v47 arch v3 (OXE 9 + LIBERO 4 mix pretrain → per-suite FT)

Pretrain base: spatial / object / goal FT from `step_100000`; libero_10 FT
from `step_95000` (split because libero_10 used a 4 GPU × accum=4 effective
batch 128 run that started earlier in the pretrain). FT recipe: `bs=8 × 2
GPU × accum=2 = eff bs 32` for the 3 short-horizon suites, `bs=8 × 4 GPU
× accum=4 = eff bs 128` for libero_10. Headless MuJoCo.

Final 4-suite numbers, **10 episodes / task** (= 100 ep total / suite):

| suite | best ckpt | 10ep success rate |
|---|---|---:|
| spatial   | step_10k / 55k / 65k / 70k (plateau) | **76 – 78 %** |
| object    | step_55k / 70k / 75k                 | **90 – 92 %** |
| goal      | step_55k / 70k / 75k                 | **87 – 89 %** |
| libero_10 | step_45k / 50k                       | **43 – 45 %** |
| **4-suite avg (10ep)** | — | **~ 74 %** |

`step_50000` ckpts (all 4 suites available and uploaded to HF, see the HF
checkpoint list below): spatial 72 / object 92 / goal 89 / libero_10 43 = avg
**74 %**. 5-ep evals run at every save tend to overshoot or undershoot by
±10 pt on the spatial / libero_10 long-tail tasks; the 10ep numbers above
are the trustworthy ones.

History (LIBERO Spatial focus only; full table in git history):

| version | architecture | best SR (eval mode) |
|---|---|---:|
| baseline (vla-gemma-4) | ResNet-18 wrist + Gemma4 + L1 | 42 % (5ep) |
| v25 | Mode B + image preproc + warmup gripper | 74 % (5ep) |
| v33 | DA-2-MLP + soft-prompt-in-LLM + LoRA r=64 (40k steps) | **94 %** (5ep) |
| v37 | OXE 9-dom pretrain → LIBERO DA-row FT (proprio shortcut) | ~80 % (5ep) |
| v44 | v41 + proper_residual + LayerScale + LoRA all-linear | 8 % (5ep) |
| v45 | v44 arch + LIBERO 4-suite mixed pretrain (13 domains) | 20 % (5ep) |
| **v47 arch v3** | every selected LLM token stream → action_head cross-attn (h_a / h_t / h_sp_per_layer); self-attn pool = x only | **76 – 78 %** (10ep, spatial) |

v47 arch v3 invariants (full details in
[`docs/architectures/v47_arch_v3.mmd`](docs/architectures/v47_arch_v3.mmd)):

- 18 action-head blocks tapping Gemma 4 E2B 35 transformer layers at **even**
  positions; LoRA all-linear `r=64 α=128` on q/k/v/o/gate/up/down.
- LLM input layout `[BOS, soft_prompt(32), scene(256), prompt(20), wrist(256),
  proprio(1), action(64), EOS]` = **631 tokens** (after_vision +
  wrist_in_llm + proprio_placeholder).
- Selected LLM token positions (scene / wrist / proprio / soft_prompt /
  action) have their per-layer Gemma hidden states sliced back into the
  action_head's cross-attn streams. Prompt tokens are real text embeddings
  (no scatter) but are also routed via `prompt_in_task_stream`.
- `h_t = scene(256) ‖ wrist(256) ‖ proprio(1) ‖ prompt(20) = 533` tokens,
  with `h_t_mask` masking padded prompt positions at the softmax.
- `soft_prompt` gets its own per-layer cross-attn stream
  (`k_soft_prompt` / `v_soft_prompt`, AQ pattern, ungated).
- `legacy_external_in_self_pool: false` forces `h_w = h_sp = p = None`, so
  the self-attn pool sees `x` only (no wrist / soft_prompt / proprio
  concatenated post-fc1).

## Continuous-gripper FT (ReBotArm hand-teach datasets)

Three single-task FTs on top of v47 step_100000, each adding a new DA row
(domain_id=13, num_domains=14, `resume_da_row_init: random` per
[CLAUDE.md DA-row rule](CLAUDE.md#da-row-init-for-ft-do-not-copy)):

| dataset | HF repo | episodes / frames | best ckpt | HF ckpt repo |
|---|---|---:|---|---|
| pick up the bottle | [`takaki99/GEM4_pick_up_bottle`](https://huggingface.co/datasets/takaki99/GEM4_pick_up_bottle) | 301 / 57k | step_30000 | [`takaki99/GEM-4-FT-bottle`](https://huggingface.co/takaki99/GEM-4-FT-bottle) |
| replace the cookie | [`takaki99/GEM4_replace_the_cookie`](https://huggingface.co/datasets/takaki99/GEM4_replace_the_cookie) | 301 / 49k | step_15000 | [`takaki99/GEM-4-FT-cookie`](https://huggingface.co/takaki99/GEM-4-FT-cookie) |
| open the jar      | [`takaki99/GEM4_open_the_jar`](https://huggingface.co/datasets/takaki99/GEM4_open_the_jar) | 208 / 44k | step_15000 | [`takaki99/GEM-4-FT-jar`](https://huggingface.co/takaki99/GEM-4-FT-jar) |

Pipeline (HF dataset → FT → HF ckpt push) is fully scripted via
[`scripts/ft_lerobot_from_hf.py`](scripts/ft_lerobot_from_hf.py) +
[`tools/push_ckpt_to_hf.py`](tools/push_ckpt_to_hf.py); see "HF dataset → FT"
below.

## Setup

This repo uses two separate uv environments under `envs/` — the host's CPU
arch determines which one to install:

| host                                  | script                          | env dir         | wheels                                  |
|---------------------------------------|---------------------------------|-----------------|-----------------------------------------|
| x86_64 Linux (training / research)    | `bash scripts/setup_x86.sh`     | `envs/x86`      | PyTorch cu128 (driver ≥ 12.6)           |
| Jetson Orin (JetPack 6 / CUDA 12.6)   | `bash scripts/setup_jetson.sh`  | `envs/jetson`   | jetson-ai-lab JP6/cu126 (sm_87, cp310)  |

Each setup script installs uv (if missing), initialises the `VLA-Adapter` and
`X-VLA` submodules, runs `uv sync --project envs/<env>`, and finishes with a
torch + Gemma4 smoke check.

After setup, every command must target the chosen env via `--project`:

```bash
uv run --project envs/x86    python scripts/train.py configs/train/<config>.yaml
uv run --project envs/jetson python scripts/serve.py ...
```

Pre-requisites the scripts do **not** handle (host-specific):

- a sibling `vla-gemma-4/` checkout providing RLDS data + baseline checkpoints
- LIBERO simulator + assets (uses `MUJOCO_GL=osmesa` for headless render)
- Hugging Face token (`uv run --project envs/<env> huggingface-cli login`) for
  Gemma4 / SigLIP

### Why two envs?

- `tensorflow-addons==0.23.0` (transitive from `dlimp`/OXE-RLDS) has no Linux
  aarch64 wheel — the Jetson env therefore omits the RLDS data-pipeline deps.
- Upstream PyTorch cu126/cu128/cu130 wheels are built for sm_90+ and crash
  with `no kernel image` at `.to('cuda')` on Orin (sm_87). The Jetson env
  pulls torch / torchvision from the jetson-ai-lab JP6 / cu126 index instead.
- The cu128 index publishes 3.10 **and** 3.11 wheels, but jetson-ai-lab only
  has cp310 wheels — both envs are pinned to Python 3.10 for parity.

## Repo layout

See [`CLAUDE.md`](CLAUDE.md) for the canonical layout + coding rules.
TL;DR:

```
src/vla_project/
  data/          # dataset → internal batch schema (RLDS, LeRobot, LIBERO, lerobot_preextracted)
  models/        # vision, language, projectors, action heads, vla_policy
  policies/      # runtime obs → action wrappers (XVLAAdapterPolicy)
  training/      # trainer, optim, schedulers, checkpoint, distributed
  evaluation/    # libero_eval, rollout, metrics
  robots/        # base / sim / lerobot I/O
  deployment/    # serve, predictors, gripper_normalizer
configs/
  train/         # one yaml per architecture revision + FT recipe
  eval/          # one yaml per (ckpt × suite × step) to evaluate
  accelerate/    # per-host yaml presets (dl42_1gpu, dl50_4gpu, ...)
scripts/
  train.py
  eval.py
  serve.py
  ft_lerobot_from_hf.py   # one-shot HF dataset → FT launcher (yaml-driven)
tools/
  convert_rebot_bottle_v3_to_v21.py  # HF v3 → v2.1 + hand-teach action synthesis
  compute_norm_stats_so101.py        # q01/q99 stats for EE-delta action + EE proprio
  extract_lerobot_frames.py          # mp4 → uint8 224×224 npy memmap (10× faster than runtime decode)
  push_ckpt_to_hf.py                 # ckpt dir → HF repo (private/public, optional optimizer)
data/converted/<task>_v21/
  data/chunk-000/episode_NNNNNN.parquet   # v2.1 layout LeRobot
  meta/{info.json,episodes.jsonl,tasks.jsonl,...}
  videos/observation.images.{front,wrist}/...mp4   # symlinks to HF cache
  frames_uint8/observation.images.{front,wrist}/episode_NNNNNN.npy
    # (T, 224, 224, 3) uint8 pre-extracted by tools/extract_lerobot_frames.py;
    # consumed by data/datasets/lerobot_preextracted_dataset.py for zero-mp4-
    # decode-overhead training. Optionally rsynced to a launch-host local SSD
    # (prep.frames.local_copy in the FT yaml) to avoid NFS read contention.
docs/architectures/                  # mermaid + svg diagrams (v32, v35, v47_arch_v3, ...)
```

## Training

Two flavors:

### (a) Raw train.py (single yaml, manual launch)

For experiments where prep is already done or you want a hand-written
config:

```bash
# Single GPU
CUDA_VISIBLE_DEVICES=0 \
  uv run python scripts/train.py configs/train/libero_spatial_v33.yaml

# Multi-GPU DDP
CUDA_VISIBLE_DEVICES=0,1,2,3 \
  uv run accelerate launch \
    --config_file configs/accelerate/dl50_4gpu.yaml \
    --main_process_port 29501 \
    scripts/train.py configs/train/<config>.yaml
```

Run names + checkpoints land under `outputs/<wandb.name>/`.

### (b) HF dataset → FT pipeline (`scripts/ft_lerobot_from_hf.py`)

End-to-end, **fully yaml-driven** launcher for the LeRobot HF flow:
HF download → v3→v2.1 convert → norm stats → 224×224 uint8 frame extract
→ optional rsync to local SSD → accelerate launch. Idempotent: each step
skips if its output exists.

```bash
# Preview (no execution)
uv run python scripts/ft_lerobot_from_hf.py configs/train/<your_ft>.yaml --dry_run

# Real run
uv run python scripts/ft_lerobot_from_hf.py configs/train/<your_ft>.yaml

# Force re-extract frames (e.g. dataset got updated)
uv run python scripts/ft_lerobot_from_hf.py configs/train/<your_ft>.yaml --force_extract

# Stop before the actual accelerate launch (prep only)
uv run python scripts/ft_lerobot_from_hf.py configs/train/<your_ft>.yaml --no_launch
```

The yaml adds two blocks beyond a normal train config; see
[`configs/train/_example_ft_from_hf.yaml`](configs/train/_example_ft_from_hf.yaml)
for the full schema. CLI flags are operational only (`--dry_run`,
`--no_launch`, `--force_convert / _stats / _extract / _local`) — all
parameters (lr, freeze, batch, GPUs, port, etc.) live in the yaml.

```yaml
prep:
  hf:
    repo_id: takaki99/GEM4_<task>
  norm_stats:
    dataset_key: <key>
  frames:
    pre_extract: true
    workers: 16
    local_copy:                     # optional, avoids NFS read contention
      enabled: true
      host: dl42
      path: /var/tmp/<key>_frames_uint8

launch:
  host: dl42                        # null = local
  cuda_visible_devices: "0,1,2,3"
  num_processes: 4
  main_process_port: 29516
  accelerate_config: configs/accelerate/dl50_4gpu.yaml
  # cuda_home: /tmp/micromamba/envs/cuda-nvcc   # set on dl41 only
```

## Evaluation

```bash
uv run python scripts/eval.py configs/eval/<your_eval>.yaml
```

Episode count per task is set in the eval yaml (`eval.num_episodes_per_task`).
Convention used for the v47 numbers above:

- **5 ep / task = 50 ep / suite** — fast sweep across many step checkpoints
  during a long FT, used to spot the ckpt ranges that look promising.
- **10 ep / task = 100 ep / suite** — pin definitive numbers; published
  results / HF-card numbers should always come from this mode. 5ep values
  drift by ±10 pt on the variance-prone tasks (spatial task_5, libero_10
  task_8, etc.).

Outputs land in `outputs/<run>/eval_step<K>[<suffix>].log` with metrics
printed inline as `[eval] metrics={...}`; per-episode MP4 videos go to
`outputs/<run>/eval_videos_step<K>[<suffix>]/`.

## Inference server

FastAPI HTTP server hosting an X-VLA-Adapter checkpoint behind MimicRec's
`POST /predict` contract. Two predictor modes:

- **`hold_position`** — emits a constant action chunk. No GPU / no
  ckpt required. Use for wire-format smoke testing.
- **`xvla_adapter`** (default) — loads a real ckpt and runs forward
  passes; returns the model's denormalized action chunk in NATIVE units.
  Requires a checkpoint dir (`meta.json` + `model.pt`).

The server returns **model-native, fully q99-denormalized action chunks**
(with optional per-checkpoint `post_process.py` applied). Contract
translation (frame conversion, gripper convention mapping, raw proprio
adaptation) is the client's responsibility. See
[`docs/superpowers/specs/2026-05-16-yamlless-hf-deploy-design.md`](docs/superpowers/specs/2026-05-16-yamlless-hf-deploy-design.md)
for details.

### HoldPosition smoke (no GPU)

```bash
uv run --project envs/x86 python scripts/serve.py \
  --predictor hold_position \
  --port 8001

curl http://127.0.0.1:8001/healthz
# {"status":"ok","predictor":"HoldPositionChunkPredictor","ready_at_ns":...}
```

### XVLAAdapter (real ckpt)

`--checkpoint` accepts either a local directory or a Hugging Face repo
id (`org/repo` or `org/repo/subfolder`). When given an HF id,
`ModelRuntime.from_export` calls `huggingface_hub.snapshot_download` and
caches under `~/.cache/huggingface/hub/`; subsequent loads are free.

```bash
# Load directly from Hugging Face Hub.
CUDA_VISIBLE_DEVICES=0 \
  uv run --project envs/x86 python scripts/serve.py \
    --checkpoint takaki99/GEM-4-FT-bottle \
    --port 8001

# To enable the ckpt-bundled post_process.py for HF-resolved ckpts,
# explicitly opt in:
CUDA_VISIBLE_DEVICES=0 \
  uv run --project envs/x86 python scripts/serve.py \
    --checkpoint takaki99/GEM-4-FT-bottle \
    --trust-checkpoint-code \
    --port 8001

# Local ckpt directory.
CUDA_VISIBLE_DEVICES=0 \
  uv run --project envs/x86 python scripts/serve.py \
    --checkpoint outputs/so101_v46_step30k_ft_dl50/checkpoints/step_2000 \
    --port 8001

curl http://127.0.0.1:8001/healthz
# {"status":"ok","predictor":"XVLAAdapterChunkPredictor","ready_at_ns":...}

# /predict body: PredictRequest schema = {image_primary, image_wrist (b64 JPEG),
#   proprio (model-input shape, already adapted by client), instruction}
# Response: {"actions": list[list[float]]}  shape (T, A) in model native units.
```

Available HF ckpts (all private under `takaki99/` except `so101-v46`):

Pretrain bases:
- [`Gemma-4-Pretrained-OXE`](https://huggingface.co/takaki99/Gemma-4-Pretrained-OXE) — v47 arch v3 OXE 9 + LIBERO 4 mixed pretrain, step_100000. Base for spatial / object / goal LIBERO FT and the ReBotArm FTs.
- [`x-vla-adapter-v47-arch-v3-step95000`](https://huggingface.co/takaki99/x-vla-adapter-v47-arch-v3-step95000) — earlier v47 snapshot, used as the base for the libero_10 FT (eff bs 128 run started before step_100000 landed).

LIBERO FT @ step_50000 (10ep SR shown):
- [`GEM-4-FT-libero-spatial`](https://huggingface.co/takaki99/GEM-4-FT-libero-spatial) — **72 %**
- [`GEM-4-FT-libero-object`](https://huggingface.co/takaki99/GEM-4-FT-libero-object) — **92 %**
- [`GEM-4-FT-libero-goal`](https://huggingface.co/takaki99/GEM-4-FT-libero-goal) — **89 %**
- [`GEM-4-FT-libero-10`](https://huggingface.co/takaki99/GEM-4-FT-libero-10) — **43 %**

Continuous-gripper FT (ReBotArm hand-teach):
- [`GEM-4-FT-bottle`](https://huggingface.co/takaki99/GEM-4-FT-bottle) — bottle FT step_30000. Bundles `gripper_normalizer.py` for `[0, 1]` post-process.
- [`GEM-4-FT-cookie`](https://huggingface.co/takaki99/GEM-4-FT-cookie) — cookie FT step_15000.
- [`GEM-4-FT-jar`](https://huggingface.co/takaki99/GEM-4-FT-jar) — jar FT step_15000.

Legacy:
- [`takaki99/so101-v46/step_2000`](https://huggingface.co/takaki99/so101-v46/tree/main/step_2000) — early SO101 FT checkpoint, not currently used. Kept as the example in the inference-server walkthrough below; for current work, point `--checkpoint` at one of the GEM-4 FT repos above instead.

A minimal smoke client lives nowhere yet; the test harness in
`tests/deployment/` exercises both predictor paths and is the easiest
reference for building a request.

**Note (schema drift):** The xvla_adapter predictor path against
`takaki99/GEM-4-FT-bottle` is currently blocked by a pre-existing schema
drift in `VLAPolicyConfig` (4 fields not yet accepted by the current
Pydantic model: `prompt_in_task_stream`, `proprio_in_task_stream`,
`soft_prompt_as_cross_attn_stream`, `legacy_external_in_self_pool`). This
is unrelated to the yamlless refactor; the hold_position smoke path works
fine. Tracked as a follow-up.

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

### LeRobot HF dataset → FT → deploy walkthrough

The full path (HF dataset → v2.1 conversion → norm stats → frame extract →
FT → push HF → serve) is automated. Three options, from most-manual to
fully-scripted:

#### Option A — One-shot launcher (recommended)

[`scripts/ft_lerobot_from_hf.py`](scripts/ft_lerobot_from_hf.py) does
everything from a single yaml:

```bash
# 1. Copy the example yaml + edit prep.hf.repo_id, dataset_key, domain_id, ...
cp configs/train/_example_ft_from_hf.yaml configs/train/<your_ft>.yaml
$EDITOR configs/train/<your_ft>.yaml

# 2. Dry-run to inspect the plan
uv run python scripts/ft_lerobot_from_hf.py configs/train/<your_ft>.yaml --dry_run

# 3. Real run
uv run python scripts/ft_lerobot_from_hf.py configs/train/<your_ft>.yaml
```

#### Option B — Tools chained manually (when you need fine control)

```bash
# 1. v3 → v2.1 conversion (synthesizes action.ee_pos from obs[t+1] for
#    hand-teach datasets that only have action.joint_pos)
uv run python tools/convert_rebot_bottle_v3_to_v21.py \
  --repo_id takaki99/GEM4_<task> \
  --out_root data/converted/<task>_v21

# 2. Q99 stats (EE-delta action with SO(3) logmap, EE proprio with /100
#    gripper normalization)
uv run python tools/compute_norm_stats_so101.py \
  --converted_root data/converted/<task>_v21 \
  --dataset_key <task_key> \
  --output data/norm_stats/<task_key>.json

# 3. Pre-extract 224×224 uint8 frames (much faster than mp4 decode at
#    train time; OpenCV + ProcessPool)
uv run python tools/extract_lerobot_frames.py \
  --root data/converted/<task>_v21 \
  --out  data/converted/<task>_v21/frames_uint8 \
  --workers 16

# 4. (Optional) rsync frames to launch host's local SSD to avoid NFS
#    contention from many dataloader workers
ssh <host> "mkdir -p /var/tmp/<task>_frames && \
    rsync -a /misc/.../<task>_v21/frames_uint8/ /var/tmp/<task>_frames/"

# 5. Write a train config yaml referencing the above paths, then launch.
#    See configs/train/bottle_pick_v47_step100k_ft_dl41_2gpu.yaml for a
#    full example.
CUDA_VISIBLE_DEVICES=0,1,2,3 \
  uv run accelerate launch \
    --config_file configs/accelerate/dl50_4gpu.yaml \
    --main_process_port 29515 \
    scripts/train.py configs/train/<your_ft>.yaml
```

#### Option C — Push a finished ckpt to HF

[`tools/push_ckpt_to_hf.py`](tools/push_ckpt_to_hf.py) — CLI wrapper around
`huggingface_hub.upload_folder` with idempotent repo creation, optional
optimizer skip, dry-run preview, token-role guard.

```bash
# preview only
uv run python tools/push_ckpt_to_hf.py <ckpt_dir> <repo_id> --dry-run

# upload model.pt + meta.json only (inference-ready, ~11 GB)
uv run python tools/push_ckpt_to_hf.py <ckpt_dir> <repo_id>

# upload everything including optimizer.pt (resume_full-capable, ~13 GB)
uv run python tools/push_ckpt_to_hf.py <ckpt_dir> <repo_id> --include-optimizer

# add gripper_normalizer.py or other deploy-side files in the ckpt dir
# before pushing; HF dedups so re-pushing model.pt is free.
```

#### Gripper normalization at deploy time

Continuous-gripper FTs (bottle / cookie / jar / so101) train with
`mask=False` on action dim 6 (raw passthrough), divided by 100 in the
dataset class. At deploy time you typically want `[0, 1]` for the robot:

```python
from vla_project.deployment.gripper_normalizer import (
    normalize_gripper, denormalize_gripper, GripperNormalizer,
)

pred_action = policy(batch)            # (B, T, 7), gripper at dim 6
grip_01 = normalize_gripper(pred_action[..., 6], raw_min=-6.0, raw_max=0.0)
#         ↑ inside × 100 to go back to raw gripper_pos, then [-6, 0] → [0, 1]
```

The `[raw_min, raw_max]` range depends on the dataset; check each
`data/norm_stats/<key>.json` `action.q01` / `q99` for dim 6 (the raw range
is q01 × 100 to q99 × 100). For bottle, `[-6, 0]` covers the trained
distribution with light clipping on extremes.

## Configuration model

Every architecture revision is a config file, not scattered code edits.

- **Train** — `configs/train/libero_<suite>_v<N>.yaml`
  fields: `model.*` (arch flags), `vision.*`, `language.*`, `data.*`,
  `train.*` (lr, freeze schedule, lr_coefs per param group), `wandb.*`.
- **Eval** — `configs/eval/libero_v<N>_step<K>.yaml` references a trained
  checkpoint by directory + step.

Architecture diffs over time live in [`docs/architectures/*.mmd`](docs/architectures).

## Known quirks

- **Two envs (`envs/x86`, `envs/jetson`)** — see [Setup](#setup) above.
  The PyTorch wheel index, Python version, and RLDS opt-in differ per env;
  always pass `--project envs/<env>` to uv commands.
- **transformers ≥ 5.0** required for Gemma4 `model_type` registration;
  lerobot's `<5.0` cap is bypassed via `[tool.uv] override-dependencies`
  in each env pyproject.
- **ROS2 on `PYTHONPATH`** breaks pytest plugin discovery — prefix tests
  with `PYTHONPATH=""`. See [`DEVELOPMENT.md`](DEVELOPMENT.md).
- **dl41 missing `/usr/local/cuda`** — accelerate's `unwrap_model` probes
  DeepSpeed which calls `installed_cuda_version`, which needs `CUDA_HOME`
  to point at a valid CUDA toolkit. Workaround: pass
  `CUDA_HOME=/tmp/micromamba/envs/cuda-nvcc` in the launch env (or set
  `launch.cuda_home` in the ft-from-hf yaml).
- **NFS read contention on multi-GPU FT** — when each rank spawns multiple
  data workers all hitting NFS-mounted `frames_uint8` npy files, one rank
  often drops to ~50% GPU util. Use `prep.frames.local_copy` in the yaml
  (or manually rsync frames to a host-local SSD) for full util.
- **dl40 EGL** broken post-2026-05-04 apt upgrade (kernel mismatch). Eval
  on dl42 instead (NFS-mounted same paths). dl40 training still works.

## Development

```bash
PYTHONPATH="" uv run --project envs/x86 pytest -v        # tests (use x86 env)
uv run --project envs/x86 ruff check src/ tests/         # lint
```

See [`DEVELOPMENT.md`](DEVELOPMENT.md) and [`CLAUDE.md`](CLAUDE.md).
