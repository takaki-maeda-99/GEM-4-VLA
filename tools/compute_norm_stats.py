"""Compute BOUNDS_Q99 action stats from a LeRobot LIBERO dataset and write JSON.

The output JSON matches the project schema consumed by
`vla_project.data.normalization.load_q99_stats`:

  {
    "<dataset_key>": {
      "action": {
        "q01":  [..., A floats],
        "q99":  [..., A floats],
        "mask": [..., A bools],
        "mean": [..., A floats],
        "std":  [..., A floats],
        "min":  [..., A floats],
        "max":  [..., A floats]
      }
    }
  }

The mean / std / min / max blocks are recorded for forward compatibility (the
legacy dataset_statistics.json includes them) but the project consumes only
q01 / q99 / mask via `load_q99_stats`.

Usage (from repo root):

  PYTHONPATH="" uv run python tools/compute_norm_stats.py \\
    --repo_id lerobot/libero_spatial_image \\
    --dataset_key libero_spatial_no_noops \\
    --output data/norm_stats/libero_spatial.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path
from typing import List, Optional

import numpy as np
import pyarrow.parquet as pq
from tqdm import tqdm

from vla_project.data.normalization import compute_q99_stats


# LIBERO single-arm Franka: 6 EE delta dims + 1 gripper binary; gripper = mask=False.
_LIBERO_DEFAULT_MASK: List[bool] = [True] * 6 + [False]


def _collect_actions(repo_id: str, episodes: Optional[List[int]]) -> np.ndarray:
    """Read action columns directly from the LeRobot dataset's parquet files.

    Bypasses `LeRobotDataset.__init__` and its `check_timestamps_sync` (which
    trips on the v2.0→v3.0 converted libero datasets) and avoids LeRobot's
    per-frame `__getitem__` overhead. Stats only need the raw `action` column.

    Args:
        repo_id: LeRobot HF dataset repo id (e.g. ``lerobot/libero_spatial_image``).
        episodes: optional subset of episode indices (filters frames by the
            ``episode_index`` column).
    """
    from huggingface_hub import snapshot_download

    local_dir = snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        allow_patterns=["data/**/*.parquet", "meta/**"],
    )
    files = sorted(glob.glob(os.path.join(local_dir, "data", "**", "*.parquet"), recursive=True))
    if not files:
        raise RuntimeError(f"no parquet files for {repo_id} under {local_dir}/data")

    cols = ["action", "episode_index"] if episodes is not None else ["action"]
    ep_filter = set(int(e) for e in episodes) if episodes is not None else None

    rows: List[np.ndarray] = []
    for f in tqdm(files, desc=f"reading {repo_id}"):
        t = pq.read_table(f, columns=cols)
        a = np.asarray(t["action"].to_pylist(), dtype=np.float32)
        if ep_filter is not None:
            ep = np.asarray(t["episode_index"].to_pylist(), dtype=np.int64)
            keep = np.isin(ep, list(ep_filter))
            a = a[keep]
        if a.size:
            rows.append(a)
    if not rows:
        raise RuntimeError(f"no frames found in {repo_id} (episodes={episodes!r})")
    return np.concatenate(rows, axis=0)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--repo_id", required=True, help="LeRobot HF dataset repo id")
    p.add_argument("--dataset_key", required=True, help="Key under which to store the action block in the output JSON")
    p.add_argument("--output", required=True, type=Path, help="Path to write the JSON")
    p.add_argument("--episodes", type=int, nargs="*", default=None, help="Subset of episode indices (default: all)")
    p.add_argument("--mask", type=int, nargs="*", default=None,
                   help="Per-dim mask as 0/1 ints (default: LIBERO single-arm "
                        "[1,1,1,1,1,1,0])")
    args = p.parse_args()

    arr = _collect_actions(args.repo_id, args.episodes)
    mask: List[bool]
    if args.mask is None:
        mask = list(_LIBERO_DEFAULT_MASK)
    else:
        mask = [bool(x) for x in args.mask]
    if arr.shape[1] != len(mask):
        raise ValueError(
            f"action dim {arr.shape[1]} != mask length {len(mask)}; pass --mask"
        )

    stats = compute_q99_stats(arr, mask=mask)
    payload = {
        args.dataset_key: {
            "action": {
                "q01":  stats.q01.tolist(),
                "q99":  stats.q99.tolist(),
                "mask": [bool(b) for b in stats.mask.tolist()],
                "mean": np.mean(arr, axis=0).astype(float).tolist(),
                "std":  np.std(arr, axis=0).astype(float).tolist(),
                "min":  np.min(arr, axis=0).astype(float).tolist(),
                "max":  np.max(arr, axis=0).astype(float).tolist(),
            }
        }
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2))
    print(f"[compute_norm_stats] wrote {args.output} (N={arr.shape[0]}, A={arr.shape[1]})")


if __name__ == "__main__":
    main()
