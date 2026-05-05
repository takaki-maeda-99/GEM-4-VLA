# Restore Plan: v25 Baseline Oracle to X-VLA-Adapter

This document tracks how to move from the stable v25 baseline-compatible path
back toward the intended X-VLA-Adapter architecture without losing the ability
to diagnose regressions.

## Ground Rules

- Keep `model.compat_profile: vla_gemma4_baseline` as the oracle path.
- Before each architecture experiment, run the v25 one-step smoke in
  `configs/train/libero_spatial_v25_smoke.yaml`.
- Change one architectural axis per experiment unless the axes are inseparable.
- For every step, record:
  - one-step smoke loss / completion,
  - short train loss curve,
  - closed-loop LIBERO result when applicable.
- Do not delete baseline-compatible code until the intended path has its own
  reliable train/eval oracle.

## Current Oracle

See `docs/baseline_compat_v25.md`.

Known reference:

| Config | Checkpoint | Closed-loop |
|--------|------------|-------------|
| `configs/eval/libero_v25_step7500.yaml` | `outputs/libero_spatial_v25/checkpoints/step_7500` | 33/50 = 66% |

## Difference Table

| Axis | v25 baseline oracle | Intended X-VLA-Adapter | Restore risk | First experiment |
|------|---------------------|------------------------|--------------|------------------|
| Environment | vla-gemma-4 venv + prismatic + LIBERO tree | Project-owned env, baseline deps only where needed | High | Keep vla-gemma-4 env until data/vision replacements are validated |
| Data source | vla-gemma-4 `modified_libero_rlds` | Internal schema fed by LeRobot/RLDS/OXE loaders | High | Build an apples-to-apples RLDS-vs-LeRobot batch diff before switching |
| Vision backend | `vision.type: timm` prismatic SigLIP | Project default HF SigLIP, or explicit backend choice | High | Train a short run with HF only after projectors are not baseline MLPs |
| Packer layout | `[BOS][prompt][scene][PROPRIO][action][EOS]` | `[BOS][soft][scene][prompt][wrist][action][EOS]` per arch card | High | Add config-gated packer layouts and shape tests before training |
| Vision placeholders | `unused_range` distinct PLE IDs | `image_token` or explicit X-VLA placeholder ranges | Medium | Compare PLE/input_ids only; then 1-step smoke |
| LLM layer taps | action head reads Gemma layers 1..24 | action head can read selected layers from 1..35 | Low | `action_head_layer_mode: even` with 24 blocks |
| Blocks | 24 action-head blocks | maybe still 24; 35 only if justified | Medium | Do not deepen head until layer-tap experiments finish |
| Projectors | baseline MLP scene/proprio, non-domain-aware | `DomainAwareLinear` scene/wrist/proprio/action decoder | High | Reintroduce DomainAwareLinear on single-domain first |
| Action decoder | head fc2 emits 7-dim action directly | head emits hidden dim, external domain-aware decoder emits action | Medium | Pair with DomainAwareLinear projector experiment |
| Soft prompt | disabled | per-domain soft prompts enabled | Medium | Enable only after single-domain DomainAware path is stable |
| Action queries | frozen zero in Mode B | trainable shared queries | Medium | Unfreeze AQ after decoder/projector path is stable |
| LLM update | frozen no-grad Mode B | Stage 1 frozen, Stage 2 optional LoRA | Medium | Keep no-grad until memory/perf measured; then LoRA short run |
| Wrist signal | per-layer wrist bridge, final wrist projection skipped | wrist tokens in intended task stream or head pool | High | Preserve wrist bridge initially; remove later as its own ablation |
| Action format | LIBERO native 7-dim, chunk 8 | X-VLA EE6D 20-dim, 30 anchors / 4s | Very high | Only after native 7-dim intended-path train/eval is stable |
| Multi-domain | single LIBERO-Spatial | LIBERO multi-suite, later OXE-style hetero data | Very high | Only after single-domain intended path is stable |

## Staged Experiments

### Stage 0: Freeze Oracle

Goal: Keep a cheap regression check for v25.

Status:

- `model.compat_profile: vla_gemma4_baseline` added.
- `configs/train/libero_spatial_v25_smoke.yaml` added.
- One-step smoke completes.

Next:

- Add a matching eval smoke config with `num_episodes_per_task: 1` for fast
  closed-loop sanity.

### Stage 1: Single-Domain Intended Shape, Native Actions

Goal: Move model shape toward X-VLA while keeping LIBERO native 7-dim actions
and the stable RLDS data source.

Proposed sequence:

1. `v25_even_layer_taps`
   - Keep the 24-block action head.
   - Set `action_head_layer_mode: even` so the 24 blocks read Gemma layers
     sampled across 1..35 instead of 1..24.
   - Keep wrist bridge unchanged (`num_blocks=24`, so SigLIP bridge remains
     valid).
   - Purpose: test deeper LLM conditioning without increasing ActionHead depth.

2. `v25_domainaware_single`
   - `compat_profile: x_vla_adapter`
   - `num_domains: 1`
   - `use_baseline_projectors: false`
   - Head returns hidden dim; external `DomainAwareLinear` decoder active.
   - Keep timm + RLDS + prompt-first layout.
   - Purpose: restore project-owned projector/decoder boundary.

3. `v25_trainable_aq`
   - `freeze_llm_and_aq: false` only if memory is acceptable.
   - Or introduce a narrower flag later if we want AQ trainable while LLM
     forward remains no-grad.
   - Purpose: restore action query learning without changing data.

4. `v25_soft_prompt_single`
   - Enable `use_soft_prompt: true` with `num_domains: 1`.
   - Purpose: restore soft prompt plumbing before multi-domain.

Exit criteria:

- Smoke train passes.
- Short train does not collapse to dataset mean.
- Closed-loop on LIBERO-Spatial is nonzero and diagnostically interpretable.

### Stage 2: Input Layout Restore

Goal: Move packer toward the architecture card.

Proposed sequence:

1. Add explicit config for `input_layout`.
   - `baseline_prompt_first`: current v25 layout.
   - `xvla_soft_scene_text_wrist_action`: intended layout.

2. Add shape/index tests for both layouts.

3. Train/eval the intended layout while keeping native 7-dim actions.

Exit criteria:

- Packer tests cover every placeholder index.
- One-step smoke passes.
- Short train loss behaves similarly enough to diagnose.

### Stage 3: Data Generalization

Goal: Stop depending on vla-gemma-4 RLDS as the only trusted loader.

Proposed sequence:

1. Add a batch-diff tool comparing:
   - vla-gemma-4 RLDS sample,
   - project LeRobot/RLDS sample,
   for image tensor, prompt ids, proprio, target action, masks.

2. Fix loader/transform differences until deltas are intentional.

3. Switch single-domain train to project-owned loader.

Exit criteria:

- Loader diff is documented.
- Native 7-dim single-domain training remains nondegenerate.

### Stage 4: X-VLA Action Alignment

Goal: Restore X-VLA's common action space.

Use `docs/superpowers/plans/2026-05-01-action-alignment.md` as the design
source.

Sequence:

1. Native 7-dim intended model remains the control.
2. Enable EE6D 20-dim, 30-anchor data path.
3. Verify loss components and inverse decode.
4. Closed-loop LIBERO eval.

Exit criteria:

- EE6D smoke train passes.
- Closed-loop policy can decode actions without safety/pathology issues.

### Stage 5: Multi-Domain

Goal: Restore the original X-VLA reason for domain-aware modules.

Sequence:

1. LIBERO-Spatial + Goal only.
2. Add Object.
3. Add LIBERO-10.
4. Later OXE-style heterogeneous datasets.

Exit criteria:

- Per-domain sampling and normalization are explicit.
- Per-domain eval does not hide one-domain collapse behind aggregate metrics.

## Immediate Next Implementation

Add a fast eval smoke config for the oracle:

- Based on `configs/eval/libero_v25_step7500.yaml`.
- `num_episodes_per_task: 1`
- optional short task subset for local smoke.
- No architecture changes.

Then run `configs/train/libero_spatial_v25_even_layers_smoke.yaml` and compare
against `configs/train/libero_spatial_v25_smoke.yaml`.
