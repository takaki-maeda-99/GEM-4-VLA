# X-VLA Action Alignment + Intention Abstraction (Plan 11 / draft)

> Draft — full TDD breakdown written when we approach implementation. The
> current Stage 2 LoRA r=64 baseline (PID 809756, wandb run `6wx7iroi`)
> finishes first to give us a 7-dim native baseline for comparison.

**Goal:** Replace LIBERO's native 7-dim delta-EE action with X-VLA's common 20-dim representation (`[abs_xyz, rot6d, gripper, 0×10]`) and predict 30 anchor poses over a 4-second window instead of dense per-frame chunks. Unblocks multi-dataset mixed training and (per X-VLA paper) reduces the brittleness of fitting human-demo micro-jitter.

## Why

1. **Action alignment** — when we mix LIBERO-Spatial / Goal / Object / 10 (already done) and later add OXE-style heterogeneous datasets, `action[0]` semantics diverge. A common 20-dim EE6D space puts every dataset in the same coordinates so gradients add coherently.
2. **Intention abstraction** — predicting 30 anchors over 4 seconds (vs 8 per-frame chunks) is a coarser-grained target. The model learns "where the EE will be" rather than "what the joystick said this exact frame", smoothing out demonstration noise.

## Architecture

### 20-dim action layout (X-VLA convention)

```
[xyz (3)] + [rot6d (6)] + [gripper (1)] + [zero padding (10)] = 20
```

- `xyz`: absolute end-effector position in robot base frame.
- `rot6d`: first two columns of rotation matrix, flattened (Zhou et al). 6-d, smooth.
- `gripper`: scalar in {0, 1} (closed / open).
- padding: reserved for the bimanual case (right arm xyz+rot6d). Single-arm fills with zeros.

### Loss split

- `pos_loss = mse(pred[:3], target[:3])`
- `rot_loss = mse(pred[3:9], target[3:9])` (geodesic-style; rot6d MSE is smooth, well-behaved)
- `gripper_loss = bce_with_logits(pred[9], target[9])`
- padding `[10:20]` ignored via per-dim weight = 0.

Combine: `loss = pos_loss + rot_loss + gripper_loss` (equal weight per X-VLA; weights configurable).

### Intention abstraction

Anchor offsets for an N-anchor window of T seconds at fps `f`:

```
offsets = [k * T / (N - 1) for k in range(N)]   # seconds, possibly non-integer
```

LeRobot's `delta_timestamps` accepts seconds and selects the nearest frame. For LIBERO (T=4, N=30, f=10): spacing ≈ 0.138 s ≈ 1.38 frames; LeRobot picks the closest integer frame.

`last_action_chunk` (when `mode='real'`) similarly uses negative anchor offsets.

## File Structure

**Create:**
- `src/vla_project/data/transforms/action_alignment.py` — `quat_to_rot6d`, `rot6d_to_quat`, `ee_pose_to_action20`, inverse `action20_to_ee_delta` for closed-loop control.
- `src/vla_project/data/transforms/intention.py` — `anchor_offsets(window_s, num_anchors, fps)`.
- `src/vla_project/training/losses_ee6d.py` — `ee6d_loss(pred, target)` splitting pos/rot/gripper.
- `tools/recompute_norm_stats_ee6d.py` — extends Plan 2 CLI for the 20-dim format (or just adds a new `--action_format ee6d` flag).
- `tests/test_action_alignment.py`, `tests/test_intention.py`, `tests/test_ee6d_loss.py`.
- `data/norm_stats/libero_spatial_ee6d.json` — 20-dim stats (only first 10 dims active; `[10:]` mask=False).

**Modify:**
- `src/vla_project/data/datasets/lerobot_libero_dataset.py` — add `action_format: Literal["native", "ee6d"]` and `anchor_window_s` / `num_anchors` config. When `ee6d`, fetch proprio at anchor times via `delta_timestamps`, transform per-anchor.
- `src/vla_project/data/constants.py` — add `ACTION_DIM_EE6D = 20`, `ANCHOR_WINDOW_S = 4.0`, `NUM_ANCHORS = 30`.
- `src/vla_project/models/vla_policy.py` — `VLAPolicyConfig.action_dim` and `action_chunk_len` now configurable; `loss_type` accepts `"ee6d"` to use `losses_ee6d.ee6d_loss`.
- `src/vla_project/policies/xvla_adapter_policy.py` — when `action_format=ee6d`, decode predicted 20-dim → LIBERO 7-dim delta action via `action20_to_ee_delta(pred, current_eef_pose)`.

**Configs:**
- `configs/train/libero_spatial_ee6d.yaml` — full ee6d run.
- `configs/eval/libero_smoke_ee6d.yaml` — closed-loop eval consuming ee6d head.

## Tasks (sketch — fully detailed when we start)

1. `quat_to_rot6d` / `rot6d_to_quat` (with tests; orthogonalize-on-decode).
2. `ee_pose_to_action20` / inverse + tests with synthetic EE poses.
3. `anchor_offsets` + tests.
4. Dataset extension (`action_format=ee6d`); per-frame proprio fetch at anchor times; loss masks for padding.
5. `ee6d_loss` + tests.
6. Policy integration; `loss_type=ee6d` switches to `ee6d_loss`.
7. `recompute_norm_stats_ee6d.py` + new stats JSON.
8. `XVLAAdapterPolicy` inverse decode for closed-loop control.
9. Smoke train + eval on LIBERO-Spatial.
10. Compare success_rate to the 7-dim baseline.

## Open questions / risks

- **LIBERO controller assumption**: LIBERO uses OSC-style delta-EE control. Need to verify the exact native action axes (axis-angle vs Euler delta?) — probably need to read the bddl env config.
- **Rotation discontinuity**: rot6d is smooth, but the conversion path `quat → R → rot6d` and back (after gripper update) needs care for SO(3) drift.
- **Stats recompute**: BOUNDS_Q99 on absolute xyz makes sense (operating volume bounded), but rot6d entries can swing widely; may want different normalization (e.g., per-dim mean/std with a fixed q99 = 1.0 since rot6d is bounded).
- **Window choice**: 4s × fps=10 = 40 frames available; 30 anchors with non-integer spacing means consecutive anchors map to the same frame ~10 times. LeRobot's nearest-neighbor will pick whatever it returns; check it's deterministic.
- **Inverse decode at inference**: pred[0] is abs pose at "+0s" anchor. The robot needs a delta from CURRENT pose. We have current pose from `obs["proprio"]` — this is the obvious source.

## Out of scope (future)

- Bi-manual datasets (would actually use the right-hand padding slots).
- Other rotation parametrizations (quaternion direct prediction, axis-angle).
- Diffusion-style action head (out of project scope).
