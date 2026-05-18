"""Add a native_action block to an existing meta.json.

Usage:
  uv run python tools/backfill_meta_native_action.py \\
    --ckpt /path/to/local/ckpt \\
    --units meter_axisangle_rad --frame world \\
    --gripper-kind absolute --gripper-units normalized_0_1 \\
    --gripper-closed 0.0 --gripper-open 1.0

HF push is manual: edit meta.json locally, then huggingface-cli upload
the modified file to the repo.

The tool rewrites meta.json in-place. It is idempotent: running twice
produces the same file.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def backfill_local(
    *,
    ckpt_dir: Path,
    units: str,
    frame: str,
    gripper_kind: str,
    gripper_units: str,
    gripper_closed: float,
    gripper_open: float,
) -> None:
    meta_p = Path(ckpt_dir) / "meta.json"
    if not meta_p.is_file():
        raise FileNotFoundError(f"meta.json not found at {meta_p}")
    meta = json.loads(meta_p.read_text())
    meta["native_action"] = {
        "units": units,
        "frame": frame,
        "gripper": {
            "kind": gripper_kind,
            "units": gripper_units,
            "sign": {"closed": float(gripper_closed), "open": float(gripper_open)},
        },
    }
    meta_p.write_text(json.dumps(meta, indent=2))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, type=Path,
                    help="local ckpt dir containing meta.json")
    ap.add_argument("--units", default="meter_axisangle_rad")
    ap.add_argument("--frame", required=True, choices=["world", "ee_local"])
    ap.add_argument("--gripper-kind", required=True,
                    choices=["absolute", "delta", "binary"])
    ap.add_argument("--gripper-units", required=True,
                    choices=["normalized_0_1", "signed_neg1_pos1", "binary_threshold_0p5"])
    ap.add_argument("--gripper-closed", type=float, required=True)
    ap.add_argument("--gripper-open", type=float, required=True)
    args = ap.parse_args(argv)
    backfill_local(
        ckpt_dir=args.ckpt,
        units=args.units, frame=args.frame,
        gripper_kind=args.gripper_kind, gripper_units=args.gripper_units,
        gripper_closed=args.gripper_closed, gripper_open=args.gripper_open,
    )
    print(f"wrote native_action to {args.ckpt}/meta.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
