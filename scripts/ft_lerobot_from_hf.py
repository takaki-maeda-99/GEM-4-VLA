"""End-to-end LeRobot HF dataset → v47-style policy FT launcher, fully YAML-driven.

Takes ONE yaml file. Everything (HF source, conversion paths, norm stats,
frame extraction, optional local SSD rsync, launch host / GPU layout) is in
the yaml. CLI flags are operational only: --dry_run, --force_convert,
--force_stats, --force_extract, --force_local, --no_launch.

The yaml has two extra top-level blocks beyond the usual train config
(model / data / train / wandb / etc):

  prep:
    hf:
      repo_id: takaki99/GEM4_open_the_jar
      converter: tools/convert_rebot_bottle_v3_to_v21.py    # optional
    norm_stats:
      tool: tools/compute_norm_stats_so101.py               # optional
      dataset_key: jar_open                                  # required
    frames:
      pre_extract: true
      tool: tools/extract_lerobot_frames.py                 # optional
      workers: 16                                            # optional, default 16
      local_copy:                                            # optional block
        enabled: true
        host: dl42
        path: /var/tmp/jar_frames_uint8

  launch:
    host: dl42                          # ssh target, null = local
    cuda_visible_devices: "0,1,2,3"
    num_processes: 4
    main_process_port: 29516
    accelerate_config: configs/accelerate/dl50_4gpu.yaml
    cuda_home: null                     # set on dl41 to /tmp/micromamba/envs/cuda-nvcc

Pipeline steps (idempotent; skip if output exists, override with --force_X):
  1. snapshot_download + v3 → v2.1 convert  (--force_convert)
  2. q01/q99 norm stats                      (--force_stats)
  3. 224×224 uint8 frame pre-extract         (--force_extract)
  4. optional rsync to launch.host local SSD (--force_local)
  5. accelerate launch scripts/train.py <cfg.yaml>

The standard ``data`` block in the yaml MUST reference the prep outputs,
i.e. ``data.root`` should equal the convert output dir, ``data.frames_root``
should equal the local-copy path (if enabled) or the extract output, and
``data.stats_path`` should equal the norm-stats json path. The launcher
does NOT rewrite these — the yaml is the single source of truth.

Example yaml: ``configs/train/_example_ft_from_hf.yaml`` (write one when
starting a new FT, copy from existing FT yaml + add the ``prep`` /
``launch`` blocks).

Usage:
  uv run python scripts/ft_lerobot_from_hf.py <cfg.yaml>
  uv run python scripts/ft_lerobot_from_hf.py <cfg.yaml> --dry_run
  uv run python scripts/ft_lerobot_from_hf.py <cfg.yaml> --force_extract
  uv run python scripts/ft_lerobot_from_hf.py <cfg.yaml> --no_launch
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

from omegaconf import OmegaConf


REPO_ROOT = Path(__file__).resolve().parents[1]


def _run(cmd: List[str], dry: bool) -> None:
    print(f"\n+ {' '.join(cmd)}", flush=True)
    if dry:
        return
    subprocess.run(cmd, check=True)


def _step_convert(args, prep: Any, converted: Path) -> None:
    """v3 → v2.1 convert via the HF converter tool."""
    converter = prep.hf.get("converter", "tools/convert_rebot_bottle_v3_to_v21.py")
    if converted.exists() and not args.force_convert:
        print(f"[skip] converted dir exists: {converted}")
        return
    if converted.exists():
        print(f"[force_convert] removing {converted}")
        if not args.dry_run:
            shutil.rmtree(converted)
    _run([
        sys.executable, str(REPO_ROOT / converter),
        "--repo_id", prep.hf.repo_id, "--out_root", str(converted),
    ], args.dry_run)


def _step_norm_stats(args, prep: Any, converted: Path, stats: Path) -> None:
    tool = prep.norm_stats.get("tool", "tools/compute_norm_stats_so101.py")
    if stats.exists() and not args.force_stats:
        print(f"[skip] norm stats exists: {stats}")
        return
    stats.parent.mkdir(parents=True, exist_ok=True)
    _run([
        sys.executable, str(REPO_ROOT / tool),
        "--converted_root", str(converted),
        "--dataset_key", str(prep.norm_stats.dataset_key),
        "--output", str(stats),
    ], args.dry_run)


def _step_extract(args, prep: Any, converted: Path) -> Path:
    tool = prep.frames.get("tool", "tools/extract_lerobot_frames.py")
    out = converted / "frames_uint8"
    if out.exists() and any(out.iterdir()) and not args.force_extract:
        print(f"[skip] frames already extracted: {out}")
        return out
    if out.exists():
        print(f"[force_extract] removing {out}")
        if not args.dry_run:
            shutil.rmtree(out)
    _run([
        sys.executable, str(REPO_ROOT / tool),
        "--root", str(converted), "--out", str(out),
        "--workers", str(prep.frames.get("workers", 16)),
    ], args.dry_run)
    return out


def _step_local_copy(args, prep: Any, src: Path) -> Path:
    lc = prep.frames.get("local_copy", None)
    if not lc or not lc.get("enabled", False):
        return src
    host = lc.get("host", None)
    dst = Path(lc.path)
    if not args.force_local:
        print(f"[skip-or-overwrite] local frames at {host or 'local'}:{dst} (use --force_local to refresh)")
    print(f"\n+ rsync {src}/ → {host or 'local'}:{dst}/")
    if args.dry_run:
        return dst
    if host:
        subprocess.run(
            ["ssh", host, f"mkdir -p {dst} && rsync -a {src}/ {dst}/"],
            check=True,
        )
    else:
        dst.mkdir(parents=True, exist_ok=True)
        subprocess.run(["rsync", "-a", f"{src}/", f"{dst}/"], check=True)
    return dst


def _step_launch(args, launch_cfg: Any, cfg_path: Path) -> None:
    if args.no_launch:
        print("[skip] --no_launch: stopping before accelerate launch.")
        return
    output_dir = REPO_ROOT / "outputs" / cfg_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "launch.log"
    accel = launch_cfg.get("accelerate_config",
                           "configs/accelerate/dl50_4gpu.yaml"
                           if int(launch_cfg.get("num_processes", 1)) > 1
                           else "configs/accelerate/dl42_1gpu.yaml")
    env_parts = ["MALLOC_ARENA_MAX=2", "MALLOC_TRIM_THRESHOLD_=0"]
    if launch_cfg.get("cuda_home"):
        env_parts.append(f"CUDA_HOME={launch_cfg.cuda_home}")
    if launch_cfg.get("cuda_visible_devices"):
        env_parts.append(f"CUDA_VISIBLE_DEVICES={launch_cfg.cuda_visible_devices}")
    env_str = " ".join(env_parts)
    main_port = int(launch_cfg.get("main_process_port", 29500))
    cfg_rel = cfg_path.relative_to(REPO_ROOT)
    launch_cmd = (
        f".venv/bin/accelerate launch --config_file {accel} "
        f"--main_process_port {main_port} scripts/train.py {cfg_rel}"
    )
    host = launch_cfg.get("host", None)
    log_rel = log_path.relative_to(REPO_ROOT)
    if host:
        remote = (
            f"cd GEM-4-VLA && nohup env {env_str} {launch_cmd} "
            f"> {log_rel} 2>&1 & echo PID=$!"
        )
        print(f"\n+ ssh {host} '{remote}'")
        if not args.dry_run:
            subprocess.run(["ssh", host, remote], check=True)
    else:
        full = f"nohup env {env_str} {launch_cmd} > {log_path} 2>&1 & echo PID=$!"
        print(f"\n+ {full}")
        if not args.dry_run:
            subprocess.run(["bash", "-c", full], check=True)


def main(argv: List[str]) -> int:
    p = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("yaml_path", type=Path,
                   help="Train config yaml (with extra prep / launch blocks).")
    p.add_argument("--dry_run", action="store_true",
                   help="Print all commands but do not execute.")
    p.add_argument("--no_launch", action="store_true",
                   help="Stop after generating prep outputs and config; do not run accelerate.")
    p.add_argument("--force_convert", action="store_true")
    p.add_argument("--force_stats", action="store_true")
    p.add_argument("--force_extract", action="store_true")
    p.add_argument("--force_local", action="store_true")
    args = p.parse_args(argv)

    cfg = OmegaConf.load(args.yaml_path)
    yaml_path = args.yaml_path.resolve()

    if "prep" not in cfg:
        print(f"[error] yaml is missing the top-level `prep:` block. "
              f"See scripts/ft_lerobot_from_hf.py module docstring for schema.",
              file=sys.stderr)
        return 2
    if "launch" not in cfg:
        print(f"[error] yaml is missing the top-level `launch:` block.", file=sys.stderr)
        return 2

    prep = cfg.prep
    repo_id = prep.hf.repo_id
    repo_basename = repo_id.split("/")[-1]
    # Default paths derived from repo basename + dataset_key (overridable in yaml).
    converted = Path(prep.hf.get("converted_dir",
                                 f"data/converted/{repo_basename}_v21")).resolve()
    stats = Path(prep.norm_stats.get("path",
                                     f"data/norm_stats/{prep.norm_stats.dataset_key}.json")).resolve()

    print(f"[ft_lerobot_from_hf] plan:")
    print(f"  yaml         : {yaml_path}")
    print(f"  repo_id      : {repo_id}")
    print(f"  dataset_key  : {prep.norm_stats.dataset_key}")
    print(f"  converted    : {converted}")
    print(f"  stats        : {stats}")
    print(f"  pre_extract  : {prep.frames.get('pre_extract', True)}")
    lc = prep.frames.get("local_copy", None)
    if lc and lc.get("enabled", False):
        print(f"  local_copy   : {lc.get('host', 'local')}:{lc.path}")
    print(f"  launch host  : {cfg.launch.get('host', 'local')} "
          f"({cfg.launch.get('cuda_visible_devices', 'auto')}, "
          f"{cfg.launch.get('num_processes', 1)} processes)")
    print(f"  dry_run      : {args.dry_run}")

    _step_convert(args, prep, converted)
    _step_norm_stats(args, prep, converted, stats)
    nfs_frames = converted / "frames_uint8"
    if prep.frames.get("pre_extract", True):
        nfs_frames = _step_extract(args, prep, converted)
    frames_root = _step_local_copy(args, prep, nfs_frames)

    # Sanity: ensure the yaml's data.* paths match what we just produced.
    # Mismatch is fine — user may intentionally use different paths — but warn.
    expected = {
        "data.root": str(converted),
        "data.frames_root": str(frames_root),
        "data.stats_path": str(stats),
    }
    for key, want in expected.items():
        actual_str = str(Path(str(OmegaConf.select(cfg, key, default=""))).resolve()) \
            if OmegaConf.select(cfg, key, default="") else ""
        if actual_str and actual_str != want:
            print(f"  [warn] yaml {key} = {actual_str}  (prep produced {want})")

    _step_launch(args, cfg.launch, yaml_path)
    print(f"\n[ft_lerobot_from_hf] DONE.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
