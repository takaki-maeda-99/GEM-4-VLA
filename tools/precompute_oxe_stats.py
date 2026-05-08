"""Precompute per-dataset Q99 stats for v37 OXE multi-domain pretrain.

For each dataset listed in configs/train/oxe_pretrain_v37.yaml's
``data.sources``, this script:

  1. Calls ``make_dataset_from_rlds`` once. RLDS canonicalizes the action via
     OXE's per-dataset standardize_fn, then either loads cached stats from
     ``<data_dir>/<dataset_name>/<version>/dataset_statistics_<hash>.json`` or
     computes them fresh by streaming the full dataset (slow first time, fast
     after that).
  2. Writes a friendly-named copy at
     ``<data_dir>/<dataset_name>/dataset_statistics.json`` wrapped under the
     dataset_name key, so ``scripts/train.py:_build_oxe_norm_manifest`` can
     find it without knowing the RLDS hash.

Run BEFORE the smoke / production trainer launch so the trainer doesn't
spend hours computing stats per child on first batch:

    /misc/dl00/takaki/X-VLA-Adapter/.venv/bin/python \
      /misc/dl00/takaki/X-VLA-Adapter/tools/precompute_oxe_stats.py \
      --config /misc/dl00/takaki/X-VLA-Adapter/configs/train/oxe_pretrain_v37.yaml \
      2>&1 | tee /misc/dl00/takaki/X-VLA-Adapter/outputs/v37_dl/precompute_stats.log
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf
tf.config.set_visible_devices([], "GPU")

from omegaconf import OmegaConf


def to_serializable(obj):
    """Recursively convert numpy arrays/scalars to Python lists/numbers."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer, np.bool_)):
        return obj.item()
    if isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_serializable(v) for v in obj]
    return obj


def precompute_one(data_dir: Path, dataset_name: str, force: bool) -> dict:
    from prismatic.vla.datasets.rlds.dataset import make_dataset_from_rlds
    from prismatic.vla.datasets.rlds.oxe import get_oxe_dataset_kwargs_and_weights
    from prismatic.vla.datasets.rlds.utils.data_utils import NormalizationType

    friendly_path = data_dir / dataset_name / "dataset_statistics.json"
    if friendly_path.is_file() and not force:
        existing = json.loads(friendly_path.read_text())
        if dataset_name in existing and "action" in existing[dataset_name]:
            print(f"[{dataset_name}] friendly stats already present at {friendly_path}; skipping (use --force to regenerate)")
            return existing[dataset_name]

    print(f"[{dataset_name}] resolving OXE dataset_kwargs...", flush=True)
    per_dataset_kwargs, _weights = get_oxe_dataset_kwargs_and_weights(
        str(data_dir),
        [(dataset_name, 1.0)],
        load_camera_views=("primary", "wrist"),
        load_depth=False,
        load_proprio=True,
        load_language=True,
        action_proprio_normalization_type=NormalizationType.BOUNDS_Q99,
    )
    if len(per_dataset_kwargs) == 0:
        raise RuntimeError(
            f"[{dataset_name}] get_oxe_dataset_kwargs_and_weights returned empty; "
            f"is the dataset registered in OXE configs.py and present on disk?"
        )
    dk = dict(per_dataset_kwargs[0])

    print(f"[{dataset_name}] calling make_dataset_from_rlds (may compute stats on first run; takes minutes-to-hours)...", flush=True)
    t0 = time.time()
    _ds, stats = make_dataset_from_rlds(**dk, train=True)
    elapsed = time.time() - t0
    print(f"[{dataset_name}] stats ready in {elapsed:.0f}s", flush=True)

    serialized = to_serializable(stats)

    friendly_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {dataset_name: serialized}
    friendly_path.write_text(json.dumps(payload, indent=2))
    print(f"[{dataset_name}] wrote friendly stats: {friendly_path}", flush=True)
    return serialized


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to oxe_pretrain_v37.yaml")
    parser.add_argument("--only", type=str, default="", help="Single dataset_name to precompute")
    parser.add_argument("--force", action="store_true", help="Recompute even if friendly stats exist")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    data_dir = Path(str(cfg.data.data_dir))
    sources = list(cfg.data.sources)
    print(f"v37 OXE stats precompute")
    print(f"  data_dir: {data_dir}")
    print(f"  sources: {[str(s.dataset_name) for s in sources]}")

    summary = []
    for src in sources:
        name = str(src.dataset_name)
        if args.only and name != args.only:
            continue
        ds_dir = data_dir / name
        if not ds_dir.is_dir():
            print(f"[{name}] SKIP — not yet downloaded ({ds_dir} missing)")
            summary.append({"name": name, "status": "skip_missing"})
            continue
        try:
            stats = precompute_one(data_dir, name, force=args.force)
            summary.append({
                "name": name, "status": "ok",
                "num_transitions": stats.get("num_transitions"),
                "action_q01_first": stats.get("action", {}).get("q01", [None])[0],
            })
        except Exception as e:
            print(f"[{name}] FAILED: {type(e).__name__}: {str(e)[:300]}")
            summary.append({"name": name, "status": "fail", "error": str(e)[:300]})

    print("\n=== precompute summary ===")
    for s in summary:
        print(f"  {s['name']:<60s} {s['status']:<14s} {s.get('num_transitions', '')}")


if __name__ == "__main__":
    main()
