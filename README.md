# GEM-4-VLA

Wearable Vision-Language-Action assistant built on **SigLIP + Gemma-4-E2B +
per-domain projectors + L1 action head**, targeting LIBERO benchmarks and
real-robot deployment behind MimicRec.

For project motivation, the wearable system overview, the MimicRec / MimicAnno
companion tools, and the broader research context, see the Kaggle write-up.
**This README focuses on reproducing the VLA training, evaluation, and
inference pipeline shipped in this repo.**

日本語版: [README.ja.md](README.ja.md)

## Results

All numbers below are **10 episodes / task × 10 tasks = 100 ep / suite**
(headless MuJoCo, `eval.num_episodes_per_task: 10`). All four LIBERO suites
fine-tune from a single pretrain base (OXE 9 datasets + LIBERO 4-suite mix,
`step_100000`).

### LIBERO 4-suite (FT step_50000)

| suite     | success rate | HF checkpoint |
|-----------|-------------:|---|
| spatial   | **72 %** | [`takaki99/GEM-4-FT-libero-spatial`](https://huggingface.co/takaki99/GEM-4-FT-libero-spatial) |
| object    | **92 %** | [`takaki99/GEM-4-FT-libero-object`](https://huggingface.co/takaki99/GEM-4-FT-libero-object)   |
| goal      | **89 %** | [`takaki99/GEM-4-FT-libero-goal`](https://huggingface.co/takaki99/GEM-4-FT-libero-goal)       |
| 10 (long) | **43 %** | [`takaki99/GEM-4-FT-libero-10`](https://huggingface.co/takaki99/GEM-4-FT-libero-10)           |
| **avg**   | **74 %** | — |

Pretrain base: [`takaki99/GEM-4-Pretrained-OXE`](https://huggingface.co/takaki99/GEM-4-Pretrained-OXE)
(`step_100000`). FT recipe: `bs=8 × 2 GPU × accum=2 = eff bs 32` for spatial /
object / goal; `bs=8 × 4 GPU × accum=4 = eff bs 128` for libero_10.

### ReBotArm FT

Single-task fine-tunes on top of `GEM-4-Pretrained-OXE`, adding one new
per-domain row (`num_domains: 13 → 14`, `resume_da_row_init: random`; see
[CLAUDE.md DA-row rule](CLAUDE.md#da-row-init-for-ft-do-not-copy)):

| task               | dataset                                                                                              | best ckpt   | HF checkpoint                                                                            |
|--------------------|------------------------------------------------------------------------------------------------------|-------------|------------------------------------------------------------------------------------------|
| pick up the bottle | [`takaki99/GEM4_pick_up_bottle`](https://huggingface.co/datasets/takaki99/GEM4_pick_up_bottle)       | step_30000  | [`takaki99/GEM-4-FT-bottle`](https://huggingface.co/takaki99/GEM-4-FT-bottle)            |
| open the jar       | dataset on HF                                                                                        | step_15000  | [`takaki99/GEM-4-FT-jar`](https://huggingface.co/takaki99/GEM-4-FT-jar)                  |

## Setup

Two separate uv environments live under `envs/` — the host's CPU architecture
determines which one to install.

| host                                | script                          | env dir         | wheels                                  |
|-------------------------------------|---------------------------------|-----------------|-----------------------------------------|
| x86_64 Linux (training / research)  | `bash scripts/setup_x86.sh`     | `envs/x86`      | PyTorch cu128 (driver ≥ 12.6)           |
| Jetson Orin (JetPack 6 / CUDA 12.6) | `bash scripts/setup_jetson.sh`  | `envs/jetson`   | jetson-ai-lab JP6/cu126 (sm_87, cp310)  |

Each setup script installs uv (if missing), runs `uv sync --project envs/<env>`,
and finishes with a torch + Gemma-4 smoke check.

After setup, every command must target the chosen env via `--project`:

```bash
uv run --project envs/x86    python scripts/train.py configs/train/<config>.yaml
uv run --project envs/jetson python scripts/serve.py ...
```

Pre-requisites the scripts do **not** handle (host-specific):

- a sibling `vla-gemma-4/` checkout providing RLDS data + baseline checkpoints
  (required for LIBERO suite FT and OXE pretrain reproduction — the FT
  configs set `data.data_dir` to a path under this checkout. The
  eval-from-HF flow in section A below does NOT need it.)
- LIBERO simulator + assets (uses `MUJOCO_GL=osmesa` for headless render)
- Hugging Face token (`uv run --project envs/<env> hf auth login`) for
  Gemma-4 / SigLIP

### Why two envs?

- `tensorflow-addons==0.23.0` (transitive from `dlimp`/OXE-RLDS) has no Linux
  aarch64 wheel — the Jetson env therefore omits the RLDS data-pipeline deps.
- Upstream PyTorch cu126/cu128/cu130 wheels are built for sm_90+ and crash
  with `no kernel image` at `.to('cuda')` on Orin (sm_87). The Jetson env
  pulls torch / torchvision from the jetson-ai-lab JP6 / cu126 index instead.
- Both envs are pinned to Python 3.10 (jetson-ai-lab only publishes cp310).

## Reproducing the LIBERO results

The fastest path to the numbers in the Results table is to download the
published FT checkpoints from Hugging Face and run eval directly — **no GPU
training needed.** If you want to retrain from the pretrain base, jump to
**[Fine-tuning from the pretrain base](#fine-tuning-from-the-pretrain-base-optional)**
below.

### Prerequisites

- `envs/x86` set up (see [Setup](#setup) above).
- **LIBERO simulator + assets**: clone `<https://github.com/Lifelong-Robot-Learning/LIBERO>`
  somewhere local and export the path. Headless render uses `MUJOCO_GL=osmesa`.
  ```bash
  export LIBERO_PATH=/path/to/your/LIBERO
  ```
  The eval configs already reference `${LIBERO_PATH}/libero/libero/bddl_files`
  and `${LIBERO_PATH}` for task metadata + assets.
- Hugging Face token logged in (the FT ckpt repos are public, but the Gemma-4
  tokenizer fetched at init time is gated):
  ```bash
  uv run --project envs/x86 hf auth login
  ```

### A. Evaluate a published FT checkpoint (recommended)

For each LIBERO suite, download the FT ckpt into the local path the
matching eval config expects, then launch `scripts/eval.py`.

| Suite     | HF checkpoint                                                                                | Local ckpt path                                                                | Eval config                                                                                |
|-----------|----------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------|
| spatial   | [`takaki99/GEM-4-FT-libero-spatial`](https://huggingface.co/takaki99/GEM-4-FT-libero-spatial) | `outputs/libero_spatial_v47_step100k_ft_dl41_2gpu/checkpoints/step_50000`     | `configs/eval/libero_spatial_v47_step100k_ft_dl41_2gpu_step50000_10ep.yaml`                |
| object    | [`takaki99/GEM-4-FT-libero-object`](https://huggingface.co/takaki99/GEM-4-FT-libero-object)   | `outputs/libero_object_v47_step100k_ft_dl41_2gpu/checkpoints/step_50000`      | `configs/eval/libero_object_v47_step100k_ft_dl41_2gpu_step50000_10ep.yaml`                 |
| goal      | [`takaki99/GEM-4-FT-libero-goal`](https://huggingface.co/takaki99/GEM-4-FT-libero-goal)       | `outputs/libero_goal_v47_step100k_ft_dl41_2gpu/checkpoints/step_50000`        | `configs/eval/libero_goal_v47_step100k_ft_dl41_2gpu_step50000_10ep.yaml`                   |
| 10 (long) | [`takaki99/GEM-4-FT-libero-10`](https://huggingface.co/takaki99/GEM-4-FT-libero-10)           | `outputs/libero_10_v47_step95k_ft_4gpu_accum4/checkpoints/step_50000`         | `configs/eval/libero_10_v47_step95k_ft_4gpu_accum4_step50000_10ep.yaml`                    |

Example — reproducing LIBERO-Spatial (72 %):

```bash
# 1. Download the ckpt to the path the eval config expects (zero-edit reproduction).
mkdir -p outputs/libero_spatial_v47_step100k_ft_dl41_2gpu/checkpoints
uv run --project envs/x86 hf download \
  takaki99/GEM-4-FT-libero-spatial \
  --local-dir outputs/libero_spatial_v47_step100k_ft_dl41_2gpu/checkpoints/step_50000

# 2. Run eval (100 ep / suite, ~30–60 min on a single GPU).
LIBERO_PATH=/path/to/your/LIBERO \
MUJOCO_GL=osmesa \
CUDA_VISIBLE_DEVICES=0 \
  uv run --project envs/x86 python scripts/eval.py \
    configs/eval/libero_spatial_v47_step100k_ft_dl41_2gpu_step50000_10ep.yaml
```

Swap `spatial → object / goal / 10` (and the libero_10 path uses the
different run name, see table) to evaluate the other suites.

Outputs:

- `outputs/<run>/eval_step50000_10ep.log` — single-line metrics summary:
  `[eval] metrics={'success_rate': 0.72, ...}`
- `outputs/<run>/eval_videos_step50000_10ep/` — per-episode MP4s
  (100 videos / suite)

The number of episodes per task is controlled by `eval.num_episodes_per_task`
in the eval yaml. The configs above use **10 ep / task = 100 ep / suite**,
which matches the Results table. A 5 ep mode also exists for fast sweeps but
drifts by ±10 pt on variance-prone tasks (spatial task_5, libero_10 task_8).

### B. Fine-tuning from the pretrain base (optional)

If you want to retrain the FT yourself instead of using the published ckpts:

```bash
# 1. Download the pretrain base into the path FT configs' resume_ckpt expects.
mkdir -p outputs/oxe_pretrain_v47_arch_v3_libero_dl50_bs8/checkpoints
uv run --project envs/x86 hf download \
  takaki99/GEM-4-Pretrained-OXE \
  --local-dir outputs/oxe_pretrain_v47_arch_v3_libero_dl50_bs8/checkpoints/step_100000

# 2. Download the modified LIBERO RLDS data (OpenVLA's public conversion,
#    ~25 GB across 4 suites). Then point the FT configs' data.data_dir at it.
uv run --project envs/x86 hf download \
  openvla/modified_libero_rlds \
  --repo-type dataset \
  --local-dir /path/to/modified_libero_rlds
#    Either symlink the path the configs expect:
ln -s /path/to/modified_libero_rlds \
  /misc/dl00/takaki/vla-gemma-4/data/modified_libero_rlds
#    ...or edit `data.data_dir` in each FT yaml to your local path.

# 3. Launch FT for the suite of your choice.
#    Suites: libero_spatial / libero_object / libero_goal (2 GPU, eff bs 32)
CUDA_VISIBLE_DEVICES=0,1 \
  uv run --project envs/x86 accelerate launch \
    --config_file configs/accelerate/dl50_2gpu.yaml \
    --main_process_port 29501 \
    scripts/train.py \
    configs/train/libero_spatial_v47_step100k_ft_dl41_2gpu.yaml
```

libero_10 uses `configs/train/libero_10_v47_step95k_ft_4gpu_accum4.yaml`
(4 GPU, effective batch 128) — for that one, expose 4 GPUs and pass
`configs/accelerate/dl50_4gpu.yaml`. Checkpoints land under
`outputs/<wandb.name>/checkpoints/step_<N>/`.

Pretraining from scratch (OXE 9 + LIBERO 4 mix, ~100 k steps) is supported in
code but not documented here — talk to the maintainer if you need that path.

## Fine-tuning on a new LeRobot dataset (HF → FT)

End-to-end, fully yaml-driven launcher
[`scripts/ft_lerobot_from_hf.py`](scripts/ft_lerobot_from_hf.py): HF download
→ v3→v2.1 conversion → norm stats → 224×224 uint8 frame extract → optional
rsync to local SSD → accelerate launch. Each step is idempotent (skips if
output exists).

```bash
# 1. Copy the example yaml and edit prep.hf.repo_id, dataset_key, domain_id, ...
cp configs/train/_example_ft_from_hf.yaml configs/train/<your_ft>.yaml
$EDITOR configs/train/<your_ft>.yaml

# 2. Dry-run to inspect the plan (no execution)
uv run --project envs/x86 python scripts/ft_lerobot_from_hf.py \
  configs/train/<your_ft>.yaml --dry_run

# 3. Real run
uv run --project envs/x86 python scripts/ft_lerobot_from_hf.py \
  configs/train/<your_ft>.yaml
```

The yaml adds two blocks beyond a normal train config:

```yaml
prep:
  hf:
    repo_id: takaki99/GEM4_pick_up_bottle
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
```

All hyperparameters (lr, freeze, batch, etc.) live in the yaml; CLI flags are
operational only (`--dry_run`, `--no_launch`, `--force_convert / _stats /
_extract / _local`).

## Inference server

FastAPI HTTP server hosting a checkpoint behind MimicRec's `POST /predict`
contract. The server returns model-native, fully q99-denormalized action
chunks; frame conversion / gripper-convention mapping / raw-proprio adaptation
are the client's responsibility.

Two predictors:

- **`hold_position`** — emits a constant action chunk. No GPU / no ckpt
  required. For wire-format smoke testing.
- **`xvla_adapter`** (default) — loads a real ckpt and runs forward passes.

```bash
# HoldPosition smoke (no GPU)
uv run --project envs/x86 python scripts/serve.py --predictor hold_position --port 8001
curl http://127.0.0.1:8001/healthz

# Real ckpt straight from HF
CUDA_VISIBLE_DEVICES=0 \
  uv run --project envs/x86 python scripts/serve.py \
    --checkpoint takaki99/GEM-4-FT-bottle \
    --port 8001
```

`--checkpoint` accepts a local directory or an HF repo id (`org/repo` or
`org/repo/subfolder`). HF resolution caches under `~/.cache/huggingface/hub/`;
subsequent loads are free. To enable a ckpt-bundled `post_process.py` on an
HF-resolved ckpt, also pass `--trust-checkpoint-code`.

Per-request latency on a single RTX 6000 Ada with bf16 + `torch_compile: off`
is ~220 ms (budget 266 ms, logged as a warning if exceeded).

For deploy yaml authoring, gripper normalization at deploy time, the full
`POST /predict` schema, and known runtime limitations, see
[`src/vla_project/deployment/README.md`](src/vla_project/deployment/README.md).

## Repo layout

See [`CLAUDE.md`](CLAUDE.md) for the canonical layout + coding rules. TL;DR:

```
src/vla_project/
  data/          # dataset → internal batch schema (RLDS, LeRobot, LIBERO, lerobot_preextracted)
  models/        # vision, language, projectors, action heads, vla_policy
  policies/      # runtime obs → action wrappers
  training/      # trainer, optim, schedulers, checkpoint, distributed
  evaluation/    # libero_eval, rollout, metrics
  robots/        # base / sim / lerobot I/O
  deployment/    # serve, predictors, gripper_normalizer
configs/
  train/         # one yaml per architecture revision + FT recipe
  eval/          # one yaml per (ckpt × suite × step)
  accelerate/    # per-host yaml presets
scripts/
  train.py
  eval.py
  serve.py
  ft_lerobot_from_hf.py   # one-shot HF dataset → FT launcher (yaml-driven)
tools/
  push_ckpt_to_hf.py      # ckpt dir → HF repo (optional optimizer / dry-run)
  extract_lerobot_frames.py
  compute_norm_stats_so101.py
  convert_rebot_bottle_v3_to_v21.py
docs/architectures/        # mermaid diagrams (current arch + ablations)
```

Every architecture revision is a config file under `configs/train/`, not
scattered code edits. See `docs/architectures/` for the current model layout
(LLM input stream, action-head cross-attn streams, projector arrangement).

## Development

```bash
PYTHONPATH="" uv run --project envs/x86 pytest -v        # tests
uv run --project envs/x86 ruff check src/ tests/         # lint
```

See [`DEVELOPMENT.md`](DEVELOPMENT.md) and [`CLAUDE.md`](CLAUDE.md) for the
coding rules and contribution flow.

## Acknowledgments

This project builds on design and code from two upstream open-source projects:

- [**VLA-Adapter**](https://github.com/OpenHelix-Team/VLA-Adapter) (MIT) —
  the `src/prismatic/` subtree is a slimmed-down vendoring; the action-head
  block layout and per-domain projector convention also originate here.
- [**X-VLA**](https://github.com/2toinf/X-VLA) (Apache 2.0) — design
  references for the EE6D 20-dim action layout, the self-attention pool
  action-head block, the multi-domain DA-Linear projector, and the two-step
  LLM warmup curriculum.

See [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) for full attribution
and license texts.
