# Baseline Compat: v25 vla-gemma-4 LIBERO Oracle

This path is the regression oracle for the current train/eval stack. It is not
the final X-VLA-Adapter architecture; it intentionally matches the
vla-gemma-4 LIBERO baseline closely enough to validate data, training,
checkpoint, policy, and closed-loop evaluation plumbing.

## Environment

Run v25 configs with the vla-gemma-4 environment:

```bash
PYTHONPATH=/misc/dl00/takaki/X-VLA-Adapter/src:\
/misc/dl00/takaki/vla-gemma-4/VLA-Adapter:\
/misc/dl00/takaki/vla-gemma-4 \
/misc/dl00/takaki/vla-gemma-4/.venv-gemma4/bin/python ...
```

This is intentional for v25 because the baseline-compatible path depends on:

- `prismatic` RLDS data loading from vla-gemma-4.
- `prismatic.models.backbones.vision.siglip_vit.SigLIPViTBackbone` for timm
  SigLIP features.
- The vla-gemma-4 LIBERO / robosuite install and BDDL tree.
- vla-gemma-4 normalization statistics.

TensorFlow/XLA cuDNN/cuFFT/cuBLAS registration warnings can appear when the
RLDS stack initializes. They do not mean PyTorch training/eval is on CPU; check
the `[train] device=...` or `[eval] device=...` line.

## Profile

Configs that belong to this oracle set:

```yaml
model:
  compat_profile: vla_gemma4_baseline
```

`VLAPolicyConfig` enforces the baseline-compatible model shape for this profile:

- `num_domains: 1`
- `num_blocks: 24`
- `action_dim: 7`
- `action_chunk_len: 8`
- `prompt_max_len: 20`
- `use_baseline_projectors: true`
- `use_wrist_bridge: true`
- `use_soft_prompt: false`
- `freeze_llm_and_aq: true`
- `vision_placeholder_mode: unused_range`

The config must also use `vision.type: timm`; that lives outside
`VLAPolicyConfig`, so scripts/config review must keep it paired with this
profile.

## Smoke

One-step train smoke:

```bash
CUDA_VISIBLE_DEVICES=7 \
PYTHONPATH=/misc/dl00/takaki/X-VLA-Adapter/src:\
/misc/dl00/takaki/vla-gemma-4/VLA-Adapter:\
/misc/dl00/takaki/vla-gemma-4 \
WANDB_MODE=disabled \
/misc/dl00/takaki/vla-gemma-4/.venv-gemma4/bin/python \
  /misc/dl00/takaki/X-VLA-Adapter/scripts/train.py \
  /misc/dl00/takaki/X-VLA-Adapter/configs/train/libero_spatial_v25_smoke.yaml
```

Known-good smoke after the first refactor pass:

```text
[train] losses=[0.6197937726974487]
```

Closed-loop reference observed for `step_7500`:

```text
50 episodes, 33 successes, success_rate=0.66
```

Use this path to catch regressions before changing the intended X-VLA-Adapter
architecture again.
