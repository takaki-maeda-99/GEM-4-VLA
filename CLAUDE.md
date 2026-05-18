# VLA Project Structure Guidelines

This repository is for building, training, evaluating, and deploying Vision-Language-Action (VLA) models.

The codebase should prioritize:

- Clear separation between dataset, model, policy, training, evaluation, and robot I/O
- Reproducible experiments
- Stable dataset schemas
- Easy switching between simulation, offline evaluation, and real robot deployment
- Minimal coupling between neural network code and robot-specific runtime code

## Core Principle

Do not mix `model`, `policy`, and `robot` responsibilities.

```text
model    = neural network modules such as vision encoder, LLM, projector, and action head
policy   = inference wrapper that maps observation -> action
robot    = real/sim/ROS/LeRobot I/O backend
trainer  = loss, optimizer, checkpointing, logging, distributed training
dataset  = episode data -> normalized tensor batch
```

The model should not directly read ROS topics, access hardware, or perform robot control.

The robot interface should not contain training logic.

The policy is the runtime bridge between observations and actions.

## Recommended Repository Layout

```text
vla_project/
├── pyproject.toml
├── README.md
├── CLAUDE.md
│
├── configs/
│   ├── data/
│   │   ├── libero.yaml
│   │   ├── lerobot.yaml
│   │   ├── rlds.yaml
│   │   └── custom_robot.yaml
│   ├── model/
│   │   ├── openvla_lora.yaml
│   │   ├── vla_adapter.yaml
│   │   └── diffusion_policy.yaml
│   ├── train/
│   │   ├── smoke.yaml
│   │   ├── finetune.yaml
│   │   └── pretrain.yaml
│   ├── eval/
│   │   ├── libero.yaml
│   │   └── real_robot.yaml
│   └── deploy/
│       ├── local_inference.yaml
│       └── remote_server.yaml
│
├── src/
│   └── vla_project/
│       ├── data/
│       │   ├── datasets/
│       │   │   ├── lerobot_dataset.py
│       │   │   ├── rlds_dataset.py
│       │   │   ├── libero_dataset.py
│       │   │   └── custom_dataset.py
│       │   ├── transforms/
│       │   │   ├── image.py
│       │   │   ├── proprio.py
│       │   │   ├── action.py
│       │   │   └── language.py
│       │   ├── collators.py
│       │   ├── schema.py
│       │   └── normalization.py
│       │
│       ├── models/
│       │   ├── vision/
│       │   │   ├── siglip.py
│       │   │   ├── dinov2.py
│       │   │   └── resnet.py
│       │   ├── language/
│       │   │   └── llm_backbone.py
│       │   ├── projectors/
│       │   │   ├── mlp_projector.py
│       │   │   └── qformer.py
│       │   ├── action_heads/
│       │   │   ├── discrete_action_head.py
│       │   │   ├── continuous_action_head.py
│       │   │   ├── diffusion_head.py
│       │   │   └── flow_matching_head.py
│       │   └── vla_policy.py
│       │
│       ├── policies/
│       │   ├── base_policy.py
│       │   ├── openvla_policy.py
│       │   ├── vla_adapter_policy.py
│       │   └── diffusion_policy.py
│       │
│       ├── training/
│       │   ├── trainer.py
│       │   ├── losses.py
│       │   ├── optim.py
│       │   ├── schedulers.py
│       │   ├── checkpoint.py
│       │   └── distributed.py
│       │
│       ├── evaluation/
│       │   ├── metrics.py
│       │   ├── rollout.py
│       │   ├── libero_eval.py
│       │   └── real_robot_eval.py
│       │
│       ├── deployment/
│       │   ├── inference_server.py
│       │   ├── inference_client.py
│       │   ├── runtime_policy.py
│       │   └── safety_filter.py
│       │
│       ├── robots/
│       │   ├── base_robot.py
│       │   ├── ros2_robot.py
│       │   ├── lerobot_robot.py
│       │   └── sim_robot.py
│       │
│       └── utils/
│           ├── logging.py
│           ├── seed.py
│           ├── io.py
│           └── timing.py
│
├── scripts/
│   ├── train.py
│   ├── eval.py
│   ├── infer.py
│   ├── serve.py
│   ├── collect_data.py
│   ├── convert_dataset.py
│   └── visualize_episode.py
│
├── tools/
│   ├── inspect_dataset.py
│   ├── replay_episode.py
│   ├── compute_norm_stats.py
│   └── export_checkpoint.py
│
├── tests/
│   ├── test_dataset_schema.py
│   ├── test_action_normalization.py
│   ├── test_model_forward.py
│   └── test_checkpoint_load.py
│
├── notebooks/
│   └── dataset_debug.ipynb
│
├── data/
│   └── .gitkeep
│
├── outputs/
│   └── .gitkeep
│
└── checkpoints/
    └── .gitkeep
```

## Minimal Layout

For a smaller research prototype, use this reduced structure:

```text
vla_project/
├── pyproject.toml
├── configs/
│   ├── train.yaml
│   ├── model.yaml
│   └── data.yaml
├── src/vla_project/
│   ├── data/
│   │   ├── dataset.py
│   │   ├── schema.py
│   │   └── transforms.py
│   ├── models/
│   │   ├── vla.py
│   │   └── action_head.py
│   ├── policies/
│   │   └── base_policy.py
│   ├── training/
│   │   ├── trainer.py
│   │   └── losses.py
│   ├── evaluation/
│   │   └── eval.py
│   ├── deployment/
│   │   └── infer.py
│   └── robots/
│       └── base_robot.py
├── scripts/
│   ├── train.py
│   ├── eval.py
│   └── infer.py
└── tests/
    ├── test_dataset.py
    └── test_forward.py
```

## Dataset Schema

All datasets must be converted into a common internal schema before being passed to the model.

A typical batch should look like:

```python
batch = {
    "observation": {
        "image_primary": Tensor[B, T, C, H, W],
        "image_wrist": Tensor[B, T, C, H, W],
        "proprio": Tensor[B, T, D],
    },
    "language": {
        "input_ids": Tensor[B, L],
        "attention_mask": Tensor[B, L],
    },
    "action": Tensor[B, T, A],
    "action_mask": Tensor[B, T],
    "episode_id": list[str],
    "task": list[str],
}
```

Dataset-specific logic should stay inside:

```text
src/vla_project/data/datasets/
```

Common conversion, normalization, and preprocessing should stay inside:

```text
src/vla_project/data/transforms/
src/vla_project/data/schema.py
src/vla_project/data/normalization.py
```

## Action Handling

Action normalization and denormalization must be isolated from the model.

Use a dedicated module:

```text
src/vla_project/data/transforms/action.py
```

Do not scatter action scaling logic across training, evaluation, and deployment code.

The action space should be explicitly configured.

Examples:

```text
delta_ee_pose
absolute_ee_pose
joint_position
joint_velocity
gripper_open_close
base_velocity
```

Training-time and deployment-time action processing must use the same normalization statistics and action schema.

## Normalization

Store normalization statistics separately and include them in checkpoints or checkpoint metadata.

Required metadata:

```json
{
  "action_mean": [],
  "action_std": [],
  "proprio_mean": [],
  "proprio_std": [],
  "image_mean": [],
  "image_std": []
}
```

A checkpoint should not contain only model weights.

It should also preserve:

```text
- full config
- dataset version
- action schema
- normalization statistics
- tokenizer / processor settings
- model architecture settings
```

## Model Structure

The `models/` directory should contain pure neural network components.

It should not contain ROS, robot control, camera capture, or real-time runtime code.

Recommended split:

```text
models/
├── vision/
├── language/
├── projectors/
├── action_heads/
└── vla_policy.py
```

Example responsibility split:

```text
vision/        = image encoders such as SigLIP, DINOv2, ResNet
language/      = LLM backbone wrappers
projectors/    = vision/proprio/action-token projection modules
action_heads/  = continuous, discrete, diffusion, or flow-matching heads
vla_policy.py  = combined nn.Module
```

The model should generally be usable as:

```python
loss = model(batch)
```

## Policy Structure

The `policies/` directory should contain runtime inference wrappers.

A policy maps observation to action:

```python
obs = robot.get_observation()
action = policy.select_action(obs)
robot.send_action(action)
```

The policy is responsible for:

```text
- loading checkpoints
- applying preprocessors
- formatting language prompts
- calling the model
- denormalizing actions
- applying action chunking
- returning executable actions
```

The policy should not own robot hardware code directly.

## Robot Interface

Robot-specific code belongs in:

```text
src/vla_project/robots/
```

Use a common base interface:

```python
class BaseRobot:
    def connect(self): ...
    def get_observation(self) -> dict: ...
    def send_action(self, action): ...
    def reset(self): ...
    def close(self): ...
```

Concrete implementations may include:

```text
ros2_robot.py
lerobot_robot.py
sim_robot.py
```

The return value of `get_observation()` should be close to the dataset observation schema.

This reduces train/deploy mismatch.

## Deployment Runtime

Deployment-specific logic belongs in:

```text
src/vla_project/deployment/
```

In particular, use a dedicated runtime wrapper:

```text
deployment/runtime_policy.py
```

Responsibilities:

```text
- image resize/crop
- camera key mapping
- proprio formatting
- language prompt formatting
- action chunking
- action denormalization
- safety clamp
- latency measurement
- inference server/client integration
```

Do not put this logic inside the model.

## Training

Training code belongs in:

```text
src/vla_project/training/
```

Recommended files:

```text
trainer.py
losses.py
optim.py
schedulers.py
checkpoint.py
distributed.py
```

Training entrypoints in `scripts/` should be thin.

Good:

```python
# scripts/train.py
from vla_project.training.trainer import main

if __name__ == "__main__":
    main()
```

Avoid writing the full training loop directly in `scripts/train.py`.

## Evaluation

Evaluation code belongs in:

```text
src/vla_project/evaluation/
```

Separate benchmark evaluation from real robot evaluation.

Examples:

```text
libero_eval.py
real_robot_eval.py
rollout.py
metrics.py
```

Evaluation should save:

```text
- metrics
- rollout videos
- failure cases
- used config
- checkpoint reference
```

## Configuration

Use config files for data, model, training, evaluation, and deployment.

Recommended split:

```text
configs/
├── data/
├── model/
├── train/
├── eval/
└── deploy/
```

Avoid hardcoding dataset paths, camera keys, action dimensions, normalization stats, checkpoint paths, or model names in source code.

Important config fields:

```yaml
# data
dataset_type: lerobot
repo_id: user/dataset
image_keys:
  - image_primary
  - image_wrist
action_key: action
proprio_key: observation.state
horizon: 16

# model
vision_encoder: siglip
llm: gemma
projector: mlp
action_head: diffusion
use_proprio: true

# train
batch_size: 32
lr: 1e-4
precision: bf16
gradient_checkpointing: true
lora: true

# deploy
control_hz: 10
action_chunk_size: 8
inference_mode: remote_server
```

## Experiment Outputs

Each experiment should create a self-contained output directory.

Example:

```text
outputs/
└── 2026-04-29_21-00_openvla_lora_libero/
    ├── config.yaml
    ├── git_commit.txt
    ├── train.log
    ├── metrics.jsonl
    ├── norm_stats.json
    ├── action_schema.json
    ├── checkpoints/
    │   ├── step_10000/
    │   └── latest/
    └── eval/
        ├── libero_results.json
        └── videos/
```

Always save:

```text
- full resolved config
- git commit hash
- dataset version
- action schema
- normalization stats
- tokenizer / processor config
- checkpoint
- evaluation results
- rollout videos when available
```

## Scripts

Scripts should be command-line entrypoints only.

Recommended scripts:

```text
scripts/train.py
scripts/eval.py
scripts/infer.py
scripts/serve.py
scripts/collect_data.py
scripts/convert_dataset.py
scripts/visualize_episode.py
```

Keep scripts thin.

Reusable logic should live under `src/vla_project/`.

## Tools

Utility and inspection code belongs in:

```text
tools/
```

Recommended tools:

```text
inspect_dataset.py
replay_episode.py
compute_norm_stats.py
export_checkpoint.py
```

Tools may be less polished than library code, but should not duplicate core training or model logic.

## Tests

At minimum, test:

```text
- dataset schema compatibility
- action normalization / denormalization
- model forward pass
- checkpoint save/load
- policy inference path
```

Recommended tests:

```text
tests/
├── test_dataset_schema.py
├── test_action_normalization.py
├── test_model_forward.py
└── test_checkpoint_load.py
```

A small smoke test should verify that:

```text
dataset -> collator -> model.forward -> loss
```

works on a tiny batch.

## Coding Rules

* Keep model code independent from robot runtime code.
* Keep dataset conversion independent from model architecture.
* Keep normalization logic centralized.
* Keep action schema explicit.
* Keep scripts thin.
* Do not hardcode experiment-specific paths in source files.
* Do not store large datasets or checkpoints in git.
* Do not duplicate preprocessing between training and deployment.
* Prefer config-driven behavior over scattered constants.
* Save all metadata required to reproduce training and deployment.

# VLA Project Rules

## 1. Keep boundaries clear

- `data/` converts datasets into the internal batch schema.
- `models/` contains pure neural network modules.
- `policies/` converts observations into actions using a trained model.
- `robots/` handles real/sim robot I/O only.
- `training/` owns loss, optimizer, checkpointing, and logging.

Do not mix these responsibilities.

## 2. Use one internal batch schema

All datasets must be converted into the same internal schema before entering the model.

Do not let model code depend on LeRobot, RLDS, LIBERO, ROS, or custom robot-specific formats.

## 3. Describe architecture in config, not scattered code

Architecture variants must be represented by config files under `configs/model/`.

Do not hardcode architecture choices inside training scripts or model modules.

## 4. Fail fast on invalid shapes or missing fields

Do not silently reshape tensors, create dummy values, or add permissive fallbacks.

Use explicit assertions for important tensor shapes, masks, and required batch keys.

## 5. Every architecture change needs a smoke test

At minimum, verify:

dataset -> collator -> model.forward -> loss

For policy changes, verify:

observation -> policy.select_action -> executable action

# DA Row Init for FT (DO NOT COPY)

When fine-tuning with a larger `num_domains` than the pretrain checkpoint (adding new domains at FT time), `train.resume_da_row_init` MUST be `random`. **`copy_row_<n>` is forbidden.**

**Why**: OXE per-dataset proprio is canonicalized to 8-dim by zero-padding sources whose state encoding has fewer dims. Different OXE sources thus have different "zero" dims. Example measured 2026-05-10:

| dim | LIBERO (target row, e.g. row 9) std | taco_play (potential source row 1) std |
|---|---|---|
| 6 (LIBERO gripper bit) | **0.89 (most informative)** | **0.00 (zero-pad)** |

`copy_row_1` would transfer taco_play's `proprio_proj.fc1` weights — which learned to ignore dim 6 because taco_play never varies it — to the new LIBERO row, starting FT with "ignore the most informative LIBERO proprio dim". The model is severely handicapped from step 0.

The same issue applies to scene/wrist/action_decoder DA rows when distributions of relevant features differ across datasets. `random` init forces the new row to learn from scratch on the FT data without inheriting incompatible inductive biases.

# Monitoring Training Runs (READ THIS BEFORE CALLING A RUN STUCK)

The training entrypoint `scripts/train.py` calls `accelerator.log(payload, step=step)` once per optimizer step (`src/vla_project/training/trainer.py:342`). That payload routes to wandb when `wandb.enabled: true`, and is the **only** place per-step metrics (`train/loss`, `train/grad_norm`, `train/step_time_ms`, `_step`, etc.) are emitted.

**The trainer does NOT print per-step lines to stdout.** Therefore:

- After dataset init / compile completes, `launch.log` (= stdout) goes silent. This is normal, NOT a stuck state.
- "tail launch.log shows nothing new" is NOT evidence that training has stopped. It is **expected**.
- The only stdout per-step output is the `[WARN] step N: non-finite ...` NaN guard line, which fires only when forward loss or grad_norm is non-finite — typically zero events on a healthy run.

Always check wandb when assessing run progress, not just stdout:

```python
import wandb
api = wandb.Api()
run = api.run('takaki-maeda-1999-toyota-technological-institute/vla-project/<run_id>')
print(run.state, run.summary.get('_step'), run.summary.get('train/loss'))
```

The `<run_id>` is the suffix of the wandb run-dir under `/misc/dl00/takaki/GEM-4-VLA/wandb/run-<timestamp>-<run_id>/`. The wandb run also exposes `train/eta_s` and `train/progress_pct` for sanity-check on whether the loop is healthy.

A run is genuinely stuck only when ALL of:

1. wandb run.state = "running" but `_step` has not advanced for many minutes.
2. GPU util has dropped to 0% on all ranks (no compute happening).
3. No `[WARN]`, no NCCL watchdog warning, but the process is alive.

Until you have all three, the run is most likely just compiling or filling shuffle buffers — both of which are silent on stdout.

The shuffle-buffer fill IS visible in stdout (`tensorflow/core/kernels/data/shuffle_dataset_op.cc:452] Shuffle buffer filled.`). After it completes, expect ~10-30 min of silence while `torch.compile` builds the graph (longer when model code changes invalidate the inductor cache at `/tmp/torchinductor_takaki/`).

# Code Review Workflow

Use the local `codex` CLI (model: `gpt-5.5`) as a peer reviewer at the following checkpoints. Do not skip — the user has explicitly asked for this cadence.

## When to invoke

| Checkpoint | Command |
|---|---|
| After writing a design spec under `docs/superpowers/specs/` (file is untracked or staged) | `codex review --uncommitted` |
| After writing an implementation plan under `docs/superpowers/plans/` (file is untracked or staged) | `codex review --uncommitted` |
| Before each commit during implementation | `codex review --uncommitted` |
| Before opening or merging a PR | `codex review --base main` |

`codex review` cannot take an explicit file path; it reviews the diff scope (`--uncommitted`, `--base <branch>`, or `--commit <sha>`). If you need to focus codex on a specific file, isolate it as the only changed file before invoking, or pass `--title` to anchor the review header. Note that codex review **rejects** any positional `[PROMPT]` argument when used with `--uncommitted` / `--base` — pass guidance via `--title` instead.

## How to use the output

- Treat codex output as a **second opinion**, not authority. Resolve findings via the `superpowers:receiving-code-review` skill (verify each claim against the code; technical correctness wins over agreement).
- If codex flags something the human user has already approved, surface it briefly but do not silently revert the approved decision.
- Do not auto-apply codex suggestions. Read, evaluate, then decide.

## Configuration

- Default model is set in `~/.codex/config.toml` (`model = "gpt-5.5"`).
- Per-invocation override: `codex review -c model="gpt-5.5" --uncommitted`.
- If `codex` is unavailable in the environment, surface this to the user before proceeding past a checkpoint that requires it; do not silently skip.

## Per-Round Codex Audit (non-interactive, MANDATORY when active)

Separate from the diff-based `codex review` flow above. The user has explicitly asked for a **per-round** second-opinion loop on technical claims, design proposals, and audit findings — the kind of judgment that does not produce a diff but still risks being silently wrong (e.g. "the wiring is correct because…", "these safeguards are sufficient because…"). Use this whenever a round would commit you to a non-trivial direction (refactor plan, safety audit, architecture extension proposal, normalization-pipeline design).

Procedure:

1. After producing the round's analysis/proposal, write a self-contained review prompt to `/tmp/codex_review_round{N}.md`. Each prompt must:
   - Name the audited claims with file:line references.
   - Ask codex to verify each claim against the actual code, flag inaccuracies, and judge whether proposed safeguards / fixes are sufficient.
   - End with an explicit GO / GO-WITH-CHANGES / DO-NOT-PROCEED ask.
2. Run `codex exec --skip-git-repo-check < /tmp/codex_review_round{N}.md`.
3. Re-verify codex's most actionable corrections against the actual code yourself (do not relay codex verbatim — judge each point).
4. Report to the user: `✅ valid / △ partial / ❌ rebut` per codex point, with code-confirmed evidence.

Do not skip steps 3-4. Codex output is a peer review, not authority. The point of the loop is two independent reads of the same code, not delegation.