"""v37 OXE single-arm 6DOF+Gripper pretrain: download the 9 missing datasets.

Already on disk at /misc/dl00/takaki/vla-gemma-4/data/stage3_openx:
  - fractal20220817_data
  - taco_play

This script adds:
  - kuka, jaco_play, viola, berkeley_autolab_ur5, stanford_hydra,
    nyu_franka_play, austin_sailor, austin_sirius, bridge_orig

Adapted from /misc/dl00/takaki/vla-gemma-4/scripts/stage3/download_data.py.
Direct GCS copy (anonymous read of gs://gresearch/robotics/<name>/<version>/),
parallel ThreadPoolExecutor, resume by skipping existing files, status JSON.

Run with any TF 2.15 + tfds venv (we share
``/misc/dl00/takaki/vla-gemma-4/.venv-gemma4`` for that):
  /misc/dl00/takaki/vla-gemma-4/.venv-gemma4/bin/python \
    /misc/dl00/takaki/X-VLA-Adapter/tools/download_oxe_v37.py 2>&1 \
    | tee /misc/dl00/takaki/X-VLA-Adapter/outputs/v37_dl/download.log
"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import tensorflow as tf
tf.config.set_visible_devices([], "GPU")


DATA_ROOT = Path("/misc/dl00/takaki/vla-gemma-4/data/stage3_openx")
LOG_DIR = Path("/misc/dl00/takaki/X-VLA-Adapter/outputs/v37_dl")
LOG_PATH = LOG_DIR / "download.log"
STATUS_PATH = LOG_DIR / "download_status.json"

GCS_BASE = "gs://gresearch/robotics"
NUM_PARALLEL_WORKERS = 16

V37_DATASETS = [
    {"name": "jaco_play",                                                        "version": "0.1.0", "est_gb": 10,  "note": "Jaco arm, wrist=image_wrist"},
    {"name": "viola",                                                            "version": "0.1.0", "est_gb": 10,  "note": "Franka, wrist=eye_in_hand_rgb"},
    {"name": "nyu_franka_play_dataset_converted_externally_to_rlds",             "version": "0.1.0", "est_gb": 5,   "note": "Franka, no wrist"},
    {"name": "austin_sirius_dataset_converted_externally_to_rlds",               "version": "0.1.0", "est_gb": 6,   "note": "Franka, wrist=wrist_image"},
    {"name": "austin_sailor_dataset_converted_externally_to_rlds",               "version": "0.1.0", "est_gb": 18,  "note": "Franka, wrist=wrist_image"},
    {"name": "stanford_hydra_dataset_converted_externally_to_rlds",              "version": "0.1.0", "est_gb": 72,  "note": "Franka, wrist=wrist_image"},
    {"name": "berkeley_autolab_ur5",                                             "version": "0.1.0", "est_gb": 76,  "note": "UR5, wrist=hand_image"},
    {"name": "bridge",                                                           "version": "0.1.0", "est_gb": 400, "note": "WidowX (= bridge_oxe alias). bridge_orig source-website not on GCS; use OXE-format from gs://gresearch/robotics/bridge/0.1.0/"},
    # NOTE: kuka removed 2026-05-06 due to user-level disk-quota constraints
    # (would have been ~750GB; 10-domain mixture chosen over further outputs cleanup).
    # If quota is later expanded, re-add and bump v37 num_domains back to 11.
]


def log(msg: str, also_print: bool = True) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")
    if also_print:
        print(line, flush=True)


def update_status(name: str, status: dict) -> None:
    current = {}
    if STATUS_PATH.exists():
        try:
            current = json.loads(STATUS_PATH.read_text())
        except json.JSONDecodeError:
            current = {}
    current[name] = status
    STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATUS_PATH.write_text(json.dumps(current, indent=2))


def disk_size_mb(path: Path) -> float:
    if not path.exists():
        return 0.0
    total = 0
    for p in path.rglob("*"):
        if p.is_file():
            total += p.stat().st_size
    return total / 1024 / 1024


def copy_one_file(src_gs: str, dst_local: Path) -> tuple:
    dst_local.parent.mkdir(parents=True, exist_ok=True)
    if dst_local.exists():
        return (dst_local.stat().st_size, True)
    tf.io.gfile.copy(src_gs, str(dst_local), overwrite=False)
    return (dst_local.stat().st_size, False)


def download_one_dataset(spec: dict) -> dict:
    name = spec["name"]
    version = spec["version"]
    src_dir = f"{GCS_BASE}/{name}/{version}"
    dst_dir = DATA_ROOT / name / version

    log(f"=== START {name} (v{version}) ===")
    log(f"  note: {spec['note']}")
    log(f"  estimated: ~{spec['est_gb']} GB")
    log(f"  src: {src_dir}")
    log(f"  dst: {dst_dir}")
    dst_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    status = {
        "name": name, "version": version,
        "start_ts": time.strftime("%Y-%m-%d %H:%M:%S"),
        "src": src_dir, "dst": str(dst_dir),
        "status": "listing_files",
        "num_files_total": 0, "num_files_done": 0,
        "num_files_skipped_resume": 0, "bytes_done": 0,
        "elapsed_sec": 0,
    }
    update_status(name, status)

    try:
        files = tf.io.gfile.glob(f"{src_dir}/*")
        log(f"  listed {len(files)} files at {src_dir}")
        status["num_files_total"] = len(files)
        status["status"] = "copying"
        update_status(name, status)

        done, skipped, bytes_done = 0, 0, 0
        last_progress_log = time.time()
        with ThreadPoolExecutor(max_workers=NUM_PARALLEL_WORKERS) as executor:
            futures = []
            for src_f in files:
                rel = src_f.replace(f"{src_dir}/", "")
                dst_f = dst_dir / rel
                futures.append(executor.submit(copy_one_file, src_f, dst_f))
            for fut in as_completed(futures):
                size, was_skipped = fut.result()
                done += 1
                if was_skipped:
                    skipped += 1
                bytes_done += size
                if time.time() - last_progress_log > 30:
                    elapsed = time.time() - t0
                    rate_mbps = bytes_done / 1024 / 1024 / max(elapsed, 1)
                    log(f"  progress {done}/{len(files)} ({100*done/len(files):.1f}%), "
                        f"{bytes_done/1024**3:.2f} GB, {rate_mbps:.1f} MB/s, {elapsed:.0f}s elapsed")
                    status.update({
                        "num_files_done": done,
                        "num_files_skipped_resume": skipped,
                        "bytes_done": bytes_done,
                        "elapsed_sec": round(elapsed, 1),
                    })
                    update_status(name, status)
                    last_progress_log = time.time()

        elapsed = time.time() - t0
        size_mb = disk_size_mb(dst_dir)

        integrity_note = ""
        try:
            import tensorflow_datasets as tfds
            ds = tfds.load(name, data_dir=str(DATA_ROOT), split="train")
            first_ep = next(iter(ds.take(1)))
            ep_keys = list(first_ep.keys())
            integrity_note = f"load OK, first episode keys: {ep_keys[:5]}..."
        except Exception as e:
            integrity_note = f"load FAIL: {type(e).__name__}: {str(e)[:200]}"

        status.update({
            "status": "completed",
            "end_ts": time.strftime("%Y-%m-%d %H:%M:%S"),
            "num_files_done": done, "num_files_skipped_resume": skipped,
            "bytes_done": bytes_done, "elapsed_sec": round(elapsed, 1),
            "size_mb": round(size_mb, 1),
            "integrity_note": integrity_note,
        })
        log(f"  COMPLETED in {elapsed/60:.1f} min ({size_mb/1024:.2f} GB, {done} files, {skipped} resumed)")
        log(f"  integrity: {integrity_note}")
    except Exception as e:
        elapsed = time.time() - t0
        size_mb = disk_size_mb(dst_dir)
        status.update({
            "status": "failed",
            "end_ts": time.strftime("%Y-%m-%d %H:%M:%S"),
            "elapsed_sec": round(elapsed, 1),
            "size_mb_partial": round(size_mb, 1),
            "error_type": type(e).__name__,
            "error_msg": str(e)[:500],
        })
        log(f"  FAILED after {elapsed/60:.1f} min ({size_mb/1024:.2f} GB partial)")
        log(f"  error: {type(e).__name__}: {str(e)[:300]}")

    update_status(name, status)
    return status


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", type=str, default="", help="Single dataset name to download")
    parser.add_argument("--skip", type=str, default="", help="Comma-separated names to skip")
    parser.add_argument("--skip_completed", action="store_true", default=True)
    args = parser.parse_args()

    skip_list = set(args.skip.split(",")) if args.skip else set()

    log(f"v37 OXE download (direct GCS copy)")
    log(f"  storage: {DATA_ROOT}")
    log(f"  parallel workers per dataset: {NUM_PARALLEL_WORKERS}")
    log(f"  targets: {[d['name'] for d in V37_DATASETS]}")
    if skip_list:
        log(f"  user skip: {skip_list}")

    completed_names = set()
    if args.skip_completed and STATUS_PATH.exists():
        existing = json.loads(STATUS_PATH.read_text())
        completed_names = {n for n, s in existing.items() if s.get("status") == "completed"}
        if completed_names:
            log(f"  resume: skipping completed {completed_names}")

    results = []
    for spec in V37_DATASETS:
        name = spec["name"]
        if args.only and name != args.only:
            continue
        if name in skip_list:
            log(f"=== SKIP {name} (user --skip) ===")
            continue
        if name in completed_names:
            log(f"=== SKIP {name} (already completed) ===")
            continue
        res = download_one_dataset(spec)
        results.append(res)

    log("\n=== v37 OXE download summary ===")
    for res in results:
        log(f"  {res['name']:<60s} {res['status']:<12s} "
            f"{res.get('size_mb', res.get('size_mb_partial', 0))/1024:.2f} GB "
            f"({res.get('elapsed_sec', 0)/60:.1f} min)")
    total_gb = sum(res.get("size_mb", 0) for res in results) / 1024
    log(f"  TOTAL downloaded this run: {total_gb:.2f} GB")


if __name__ == "__main__":
    main()
