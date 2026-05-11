"""Compute BOUNDS_Q99 action + proprio stats for the SO101 dataset.

Reads from the already-filtered + renumbered v2.1 conversion produced by
``tools/convert_so101_v3_to_v21.py`` (success episodes only) and writes a
JSON in the schema consumed by ``vla_project.data.normalization``:

  {
    "so101_test": {
      "action":  {q01, q99, mask, mean, std, min, max},   # 7-dim EE-delta
      "proprio": {q01, q99, mask, mean, std, min, max}    # 8-dim (7 + 1 pad)
    }
  }

Action representation (matches the SO101 wrapper dataset):
  dim 0-2: action.ee_pos - observation.state.ee_pos     (delta EE position, meters)
  dim 3-5: action.ee_rotvec - observation.state.ee_rotvec (delta axis-angle, rad)
  dim 6:   action.gripper_pos / 100                       (absolute gripper, 0..1; mask=False passthrough)

Proprio representation (zero-padded to PROPRIO_DIM=8):
  dim 0-2: observation.state.ee_pos                       (m)
  dim 3-5: observation.state.ee_rotvec                    (rad)
  dim 6:   observation.state.gripper_pos / 100            (0..1)
  dim 7:   zero pad                                       (mask=False)

Mask conventions follow the LIBERO BOUNDS_Q99 convention used elsewhere in
the codebase: True ⇒ rescale (q01, q99) → (-1, 1) and clip; False ⇒
passthrough. We keep the gripper absolute and unnormalized to mirror LIBERO
(gripper dim 6 = mask=False), and the pad dim is also mask=False.

Usage:
  uv run python tools/compute_norm_stats_so101.py \
    --converted_root data/converted/takaki99_test_so101_v21 \
    --dataset_key so101_test \
    --output data/norm_stats/so101_test.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation
from tqdm import tqdm

from vla_project.data.normalization import compute_q99_stats


# Gripper raw values in the dataset are in degrees (0=closed, 100=open per
# meta/info.json gripper_convention). Rescaling to [0, 1] yields a unit-
# scale absolute gripper signal that mirrors LIBERO's binary gripper dim.
_GRIPPER_DIVISOR: float = 100.0


def _rotvec_delta(act_rot: np.ndarray, obs_rot: np.ndarray) -> np.ndarray:
    """Local SO(3) delta as a rotation vector: log(R_act @ R_obs^T).

    Plain elementwise ``act_rot - obs_rot`` on axis-angle vectors crosses
    the antipodal discontinuity at ``‖rotvec‖ = π`` and produces spurious
    ~2π jumps (codex round 2 measured 109/6303 frames with |d| > π on
    this dataset; q99 dim 4 ≈ 5.7 rad — physically impossible per-step).
    Building the relative rotation matrix and taking its log yields a
    canonical rotation vector in ``[-π, π]`` per axis with magnitudes
    matching the actual per-step angular increment (~0.05 rad mean).
    """
    R_obs = Rotation.from_rotvec(obs_rot)
    R_act = Rotation.from_rotvec(act_rot)
    return (R_act * R_obs.inv()).as_rotvec().astype(np.float32)


def _collect_ee_arrays(converted_root: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Return (action_arr, proprio_arr) over all frames of the converted ds.

    Action (7-dim per frame):
      [obs.state.ee_pos -> action.ee_pos delta (3),
       obs.state.ee_rotvec -> action.ee_rotvec delta (3),
       action.gripper_pos / 100 (1)]

    Proprio (7-dim per frame, will be zero-padded to 8 in the JSON):
      [obs.state.ee_pos (3), obs.state.ee_rotvec (3), obs.state.gripper_pos / 100 (1)]
    """
    files = sorted(glob.glob(os.path.join(converted_root, "data", "**", "*.parquet"), recursive=True))
    if not files:
        raise RuntimeError(f"no parquet files under {converted_root}/data")

    cols = [
        "observation.state.ee_pos",
        "observation.state.ee_rotvec",
        "observation.state.gripper_pos",
        "action.ee_pos",
        "action.ee_rotvec",
        "action.gripper_pos",
    ]

    actions: List[np.ndarray] = []
    proprios: List[np.ndarray] = []
    for f in tqdm(files, desc="reading parquets"):
        t = pq.read_table(f, columns=cols)
        # Each non-scalar column stores list<float32> -> to_pylist yields
        # a list of Python lists; convert via np.asarray for vectorization.
        obs_pos = np.asarray(t["observation.state.ee_pos"].to_pylist(), dtype=np.float32)
        obs_rot = np.asarray(t["observation.state.ee_rotvec"].to_pylist(), dtype=np.float32)
        obs_grip = np.asarray(t["observation.state.gripper_pos"].to_pylist(), dtype=np.float32)
        act_pos = np.asarray(t["action.ee_pos"].to_pylist(), dtype=np.float32)
        act_rot = np.asarray(t["action.ee_rotvec"].to_pylist(), dtype=np.float32)
        act_grip = np.asarray(t["action.gripper_pos"].to_pylist(), dtype=np.float32)
        if obs_grip.ndim == 1:
            obs_grip = obs_grip[:, None]
        if act_grip.ndim == 1:
            act_grip = act_grip[:, None]
        # action = EE-delta + abs gripper (rescaled to [0, 1]).
        d_pos = act_pos - obs_pos
        d_rot = _rotvec_delta(act_rot, obs_rot)
        act = np.concatenate([d_pos, d_rot, act_grip / _GRIPPER_DIVISOR], axis=1)
        prop = np.concatenate([obs_pos, obs_rot, obs_grip / _GRIPPER_DIVISOR], axis=1)
        actions.append(act)
        proprios.append(prop)

    action_arr = np.concatenate(actions, axis=0)
    proprio_arr = np.concatenate(proprios, axis=0)
    return action_arr, proprio_arr


def _stats_block(arr: np.ndarray, mask: List[bool]) -> dict:
    """Build a Q99 stats block matching ``load_q99_stats`` schema."""
    if arr.shape[1] != len(mask):
        raise ValueError(f"arr dim {arr.shape[1]} != mask length {len(mask)}")
    stats = compute_q99_stats(arr, mask=mask)
    return {
        "q01":  stats.q01.tolist(),
        "q99":  stats.q99.tolist(),
        "mask": [bool(b) for b in stats.mask.tolist()],
        "mean": np.mean(arr, axis=0).astype(float).tolist(),
        "std":  np.std(arr, axis=0).astype(float).tolist(),
        "min":  np.min(arr, axis=0).astype(float).tolist(),
        "max":  np.max(arr, axis=0).astype(float).tolist(),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--converted_root", required=True, type=Path,
                   help="Path to the v2.1-converted SO101 dataset directory.")
    p.add_argument("--dataset_key", required=True,
                   help="Top-level key under which the stats block is stored.")
    p.add_argument("--output", required=True, type=Path,
                   help="Path to write the JSON.")
    args = p.parse_args()

    action_arr, proprio_arr = _collect_ee_arrays(args.converted_root)
    print(f"action_arr: shape={action_arr.shape} dtype={action_arr.dtype}")
    print(f"proprio_arr: shape={proprio_arr.shape} dtype={proprio_arr.dtype}")

    # Pad proprio 7 -> 8 with a zero column so it matches PROPRIO_DIM=8 used
    # by the project's projector. The pad dim has mask=False (passthrough)
    # so normalization stays a no-op there.
    if proprio_arr.shape[1] != 7:
        raise AssertionError(f"expected proprio 7-dim, got {proprio_arr.shape[1]}")
    pad = np.zeros((proprio_arr.shape[0], 1), dtype=np.float32)
    proprio_padded = np.concatenate([proprio_arr, pad], axis=1)

    # Mask: gripper (dim 6) passthrough to mirror LIBERO; pad (dim 7)
    # passthrough.
    # Action gripper (dim 6) stays mask=False to mirror LIBERO's
    # _LIBERO_DEFAULT_MASK precedent. Proprio gripper (dim 6) is normalized
    # to align with the RLDS proprio convention which defaults all dims to
    # mask=True (codex round 2); only the zero-pad dim 7 is passthrough.
    action_mask = [True] * 6 + [False]
    proprio_mask = [True] * 7 + [False]

    payload = {
        args.dataset_key: {
            "action":  _stats_block(action_arr, action_mask),
            "proprio": _stats_block(proprio_padded, proprio_mask),
        }
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2))
    print(
        f"[compute_norm_stats_so101] wrote {args.output} "
        f"(action N={action_arr.shape[0]} A=7, proprio N={proprio_arr.shape[0]} P=8)"
    )


if __name__ == "__main__":
    main()
