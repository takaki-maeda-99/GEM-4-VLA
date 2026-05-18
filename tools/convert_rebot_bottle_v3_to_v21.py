"""Convert takaki99/GEM4_pick_up_bottle (LeRobot v3.0, hand-teach ReBotArm)
to v2.1 layout, dropping failed/deleted episodes.

Hand-teach corollary: the dataset records ``action.joint_pos`` (6-dim) +
``action.gripper_pos``, not ``action.ee_*``. Since a human moved the arm,
``observation.state.ee_pos[t+1]`` IS the "commanded" pose at time t. We
synthesize the SO101-style v2.1 features ``action.ee_pos`` /
``action.ee_rotvec`` by shifting the obs columns by one frame (last
frame holds at obs[T-1]). The downstream EE-delta target
``a_pos - o_pos`` then equals ``obs[t+1] - obs[t]`` — the actual
achieved delta — matching v47's 7-dim action schema.

Other diffs from convert_so101_v3_to_v21.py: default ``--repo_id`` and
``--out_root`` point at the bottle dataset.

Output layout (v2.1):
  <out_root>/
    meta/
      info.json            (codebase_version=v2.1, episode_chunk/episode_index format keys,
                            features extended to expose ee_pos/ee_rotvec/gripper_pos
                            from both observation.state.* and action.* so that
                            LeRobotDataset.delta_timestamps can chunk them)
      tasks.jsonl          (one task per line)
      episodes.jsonl       (renumbered 0..N-1, with `length` cumsum to 6303)
    data/chunk-000/episode_{new_idx:06d}.parquet   (rewritten: episode_index = new_idx, index = cumulative offset)
    videos/<video_key>/chunk-000/episode_{new_idx:06d}.mp4   (symlink to original)

Usage:
  uv run python tools/convert_rebot_bottle_v3_to_v21.py \
    --repo_id takaki99/GEM4_pick_up_bottle \
    --out_root data/converted/takaki99_GEM4_pick_up_bottle_v21
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Dict, List

import jsonlines
import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import snapshot_download


# Columns that hold EE pose / gripper data. We expose these as top-level
# features so that LeRobotDataset.delta_timestamps can fetch chunks of them.
_EE_OBS_FEATURES: Dict[str, Dict[str, Any]] = {
    "observation.state.ee_pos": {
        "dtype": "float32",
        "shape": [3],
        "names": ["x", "y", "z"],
    },
    "observation.state.ee_rotvec": {
        "dtype": "float32",
        "shape": [3],
        "names": ["rx", "ry", "rz"],
    },
    "observation.state.gripper_pos": {
        "dtype": "float32",
        "shape": [1],
        "names": ["gripper"],
    },
    "action.ee_pos": {
        "dtype": "float32",
        "shape": [3],
        "names": ["x", "y", "z"],
    },
    "action.ee_rotvec": {
        "dtype": "float32",
        "shape": [3],
        "names": ["rx", "ry", "rz"],
    },
    "action.gripper_pos": {
        "dtype": "float32",
        "shape": [1],
        "names": ["gripper"],
    },
}


def _build_v21_info(v3_info: Dict[str, Any], total_episodes: int, total_frames: int) -> Dict[str, Any]:
    info = dict(v3_info)
    info["codebase_version"] = "v2.1"
    info["total_episodes"] = total_episodes
    info["total_frames"] = total_frames
    info["total_chunks"] = 1
    info["total_videos"] = total_episodes * len(_video_keys_from_info(v3_info))
    info["splits"] = {"train": f"0:{total_episodes}"}
    info["data_path"] = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
    # v3 video layout = videos/{video_key}/chunk-XXX/episode_YYYYYY.mp4 with
    # video_key/chunk swapped vs lerobot's v2.1 default. We keep the v3 order
    # because (a) lerobot reads video_path as a format string at lookup time
    # so any layout consistent with the format string works, (b) symlinking
    # preserves the original layout cheaply.
    info["video_path"] = "videos/{video_key}/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.mp4"
    features = dict(info.get("features", {}))
    # The granular EE columns live in the parquet but are not declared as
    # features in the v3 info.json. Declare them so delta_timestamps can
    # fetch chunks of them via LeRobotDataset.
    for k, ft in _EE_OBS_FEATURES.items():
        features[k] = ft
    info["features"] = features
    # The action / observation.state in v3 info.json are 6-dim joint_pos.
    # Keep them as-is so existing parquet columns match the declared shape.
    # 'data_files_size_in_mb' / 'video_files_size_in_mb' are non-load-bearing
    # for the runtime path; drop if present to avoid stale numbers.
    info.pop("data_files_size_in_mb", None)
    info.pop("video_files_size_in_mb", None)
    return info


def _video_keys_from_info(info: Dict[str, Any]) -> List[str]:
    return [k for k, ft in info.get("features", {}).items() if ft.get("dtype") == "video"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo_id", default="takaki99/GEM4_pick_up_bottle")
    p.add_argument("--out_root", required=True,
                   help="Output directory for the v2.1-converted dataset (will be cleaned).")
    p.add_argument("--allow_failed", action="store_true",
                   help="Skip the success/deleted filter (debug).")
    args = p.parse_args()

    out = Path(args.out_root).resolve()
    if out.exists():
        print(f"removing existing out_root {out}")
        shutil.rmtree(out)
    out.mkdir(parents=True)

    print(f"snapshot_download({args.repo_id}) ...")
    src = Path(snapshot_download(
        repo_id=args.repo_id,
        repo_type="dataset",
        allow_patterns=["meta/**", "data/**", "videos/**"],
    ))

    v3_info = json.loads((src / "meta/info.json").read_text())
    video_keys = _video_keys_from_info(v3_info)
    print(f"video_keys: {video_keys}")

    eps_meta = pq.read_table(src / "meta/episodes/chunk-000/file-000.parquet").to_pandas()
    tasks = pq.read_table(src / "meta/tasks.parquet").to_pandas()
    print(f"v3: {len(eps_meta)} episodes, {len(tasks)} tasks")

    if args.allow_failed:
        ok = eps_meta.copy()
    else:
        # `deleted` column may be missing on newer hand-teach datasets. Synthesize
        # an all-False column in that case.
        if "deleted" not in eps_meta.columns:
            eps_meta = eps_meta.assign(deleted=False)
        # `deleted` column has None/False/True mixed. fillna(False).eq(True)
        # safely treats None/NaN as not-deleted under both object and
        # nullable-boolean dtypes (codex round 1).
        ok = eps_meta[
            eps_meta["success"].eq(True)
            & ~eps_meta["deleted"].fillna(False).eq(True)
        ]
    ok = ok.sort_values("episode_index").reset_index(drop=True)
    print(f"after filter: {len(ok)} episodes, {int(ok['length'].sum())} frames")

    # Renumber: original episode_index -> new contiguous 0..N-1
    new_to_orig: Dict[int, int] = {}
    cumulative_offset = 0
    new_lengths: List[int] = []
    for new_idx, row in ok.iterrows():
        orig_idx = int(row["episode_index"])
        new_to_orig[int(new_idx)] = orig_idx
        new_lengths.append(int(row["length"]))

    (out / "meta").mkdir()
    (out / "data/chunk-000").mkdir(parents=True)
    for vkey in video_keys:
        (out / "videos" / vkey / "chunk-000").mkdir(parents=True)

    # Write episodes.jsonl with renumbered indices but ORIGINAL task strings.
    with jsonlines.open(out / "meta/episodes.jsonl", "w") as w:
        for new_idx in range(len(ok)):
            row = ok.iloc[new_idx]
            raw_tasks = row["tasks"]
            # plain str is iterable (over chars), so check str FIRST. Lists,
            # numpy object arrays, tuples → iterate normally (codex round 1).
            if isinstance(raw_tasks, str):
                ep_tasks = [raw_tasks]
            else:
                ep_tasks = list(raw_tasks)
            w.write({
                "episode_index": int(new_idx),
                "tasks": [str(t) for t in ep_tasks],
                "length": int(row["length"]),
            })

    # Write tasks.jsonl
    with jsonlines.open(out / "meta/tasks.jsonl", "w") as w:
        for _, row in tasks.iterrows():
            w.write({"task_index": int(row["task_index"]), "task": str(row["task"])})

    # Copy + renumber parquets, symlink videos.
    cumulative_offset = 0
    for new_idx in range(len(ok)):
        orig_idx = new_to_orig[new_idx]
        ep_len = new_lengths[new_idx]
        src_parquet = src / "data/chunk-000" / f"episode_{orig_idx:06d}.parquet"
        dst_parquet = out / "data/chunk-000" / f"episode_{new_idx:06d}.parquet"

        # Rewrite episode_index column to new_idx and index column to a
        # contiguous cumulative offset.
        tbl = pq.read_table(src_parquet)
        df = tbl.to_pandas()
        assert len(df) == ep_len, f"length mismatch ep {orig_idx}: {len(df)} != {ep_len}"
        df["episode_index"] = int(new_idx)
        df["index"] = list(range(cumulative_offset, cumulative_offset + ep_len))
        cumulative_offset += ep_len

        # Hand-teach action synthesis: bottle dataset has no action.ee_pos /
        # action.ee_rotvec columns (only action.joint_pos + action.gripper_pos).
        # Use obs.state.ee_pos[t+1] as the commanded pose at frame t (last
        # frame holds at obs[T-1] so we never go OOB). The SO101-style
        # downstream pipeline then computes delta = a_pos - o_pos =
        # obs[t+1] - obs[t], which is the actual achieved delta.
        ep_ee_pos = df["observation.state.ee_pos"].tolist()
        ep_ee_rotvec = df["observation.state.ee_rotvec"].tolist()
        df["action.ee_pos"] = ep_ee_pos[1:] + [ep_ee_pos[-1]]
        df["action.ee_rotvec"] = ep_ee_rotvec[1:] + [ep_ee_rotvec[-1]]
        # Preserve dtypes by going back through arrow.
        out_tbl = pa.Table.from_pandas(df, preserve_index=False)
        pq.write_table(out_tbl, dst_parquet)

        # Symlink videos.
        for vkey in video_keys:
            src_mp4 = src / "videos" / vkey / "chunk-000" / f"episode_{orig_idx:06d}.mp4"
            dst_mp4 = out / "videos" / vkey / "chunk-000" / f"episode_{new_idx:06d}.mp4"
            if not src_mp4.exists():
                raise FileNotFoundError(src_mp4)
            dst_mp4.symlink_to(src_mp4)

    total_frames = cumulative_offset
    info = _build_v21_info(v3_info, total_episodes=len(ok), total_frames=total_frames)
    (out / "meta/info.json").write_text(json.dumps(info, indent=2))

    # Map of new_idx -> orig_idx for downstream tools (norm stats can use
    # this to pull the original episode metadata if needed).
    (out / "meta/episode_index_mapping.json").write_text(
        json.dumps({str(new): int(orig) for new, orig in new_to_orig.items()}, indent=2)
    )

    # codebase_version=v2.1 makes lerobot/datasets/lerobot_dataset.py
    # load_metadata() call load_episodes_stats() which directly opens
    # meta/episodes_stats.jsonl. Missing file raises FileNotFoundError and
    # the loader falls back to Hub revision resolution that fails (no
    # version tag on this repo). Our downstream pipeline uses its own Q99
    # stats elsewhere, so write minimal stubs: empty stats dicts pass
    # through aggregate_stats without error (empty data_keys).
    with jsonlines.open(out / "meta/episodes_stats.jsonl", "w") as w:
        for new_idx in range(len(ok)):
            w.write({"episode_index": int(new_idx), "stats": {}})

    print(f"done: wrote {len(ok)} episodes, {total_frames} frames to {out}")


if __name__ == "__main__":
    main()
