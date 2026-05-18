"""Pre-extract video frames from a v2.1 LeRobot dataset to uint8 npy files,
skipping mp4 decode at training time entirely.

For each (episode_index, camera_key) pair, read the mp4, decode all frames,
apply SiglipImageTransform geometric ops (Resize shorter→248 + CenterCrop 224),
and save as uint8 npy of shape [T, 224, 224, 3]. The dataset class then
memmap-loads the per-episode npy and applies only the Normalize(0.5) step at
batch time.

Usage:
  uv run python tools/extract_lerobot_frames.py \\
    --root data/converted/takaki99_GEM4_pick_up_bottle_v21 \\
    --out data/converted/takaki99_GEM4_pick_up_bottle_v21/frames_uint8 \\
    --workers 16
"""
from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np


_IMG_SIZE = 224
_RESIZE_SHORTER = 248


def _decode_resize_crop_one(video_path: str) -> np.ndarray:
    """Decode a video, return all frames as uint8 [T, 224, 224, 3].

    Mirrors ``vla_project/data/transforms/image.py SiglipImageTransform``
    geometric ops: Resize(shorter→248, bicubic) + CenterCrop(224).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"failed to open {video_path}")
    frames: List[np.ndarray] = []
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        # BGR -> RGB; resize shorter side to 248 (cubic); center crop 224.
        frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        h, w = frame.shape[:2]
        if h <= w:
            new_h = _RESIZE_SHORTER
            new_w = int(round(w * (_RESIZE_SHORTER / h)))
        else:
            new_w = _RESIZE_SHORTER
            new_h = int(round(h * (_RESIZE_SHORTER / w)))
        frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        y0 = (new_h - _IMG_SIZE) // 2
        x0 = (new_w - _IMG_SIZE) // 2
        frame = frame[y0:y0 + _IMG_SIZE, x0:x0 + _IMG_SIZE]
        frames.append(frame)
    cap.release()
    if not frames:
        raise RuntimeError(f"no frames decoded from {video_path}")
    return np.stack(frames, axis=0)


def _worker(args: Tuple[str, str]) -> Tuple[str, int, str]:
    src, dst = args
    if os.path.exists(dst):
        return src, -1, "skip"  # already exists
    arr = _decode_resize_crop_one(src)
    np.save(dst, arr)
    return src, arr.shape[0], "ok"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True, type=Path,
                   help="v2.1 LeRobot dataset root (contains videos/ subdir).")
    p.add_argument("--out", required=True, type=Path,
                   help="Output directory for per-episode npy files.")
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args()

    root = args.root.resolve()
    out = args.out.resolve()
    info = json.loads((root / "meta/info.json").read_text())
    video_keys = [k for k, ft in info["features"].items() if ft.get("dtype") == "video"]
    total_eps = int(info["total_episodes"])
    print(f"[extract] {total_eps} episodes × {len(video_keys)} cameras = "
          f"{total_eps * len(video_keys)} videos to decode")

    tasks: List[Tuple[str, str]] = []
    for vkey in video_keys:
        (out / vkey).mkdir(parents=True, exist_ok=True)
        for ep in range(total_eps):
            src = root / "videos" / vkey / "chunk-000" / f"episode_{ep:06d}.mp4"
            dst = out / vkey / f"episode_{ep:06d}.npy"
            if not src.exists():
                raise FileNotFoundError(src)
            tasks.append((str(src), str(dst)))

    print(f"[extract] dispatching {len(tasks)} jobs to {args.workers} workers")
    done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(_worker, t) for t in tasks]
        for fut in as_completed(futures):
            src, n, status = fut.result()
            done += 1
            if done % 50 == 0 or done == len(tasks):
                print(f"  [{done}/{len(tasks)}] {status} {Path(src).name} (T={n})")
    print(f"[extract] DONE: wrote npy files to {out}")


if __name__ == "__main__":
    main()
