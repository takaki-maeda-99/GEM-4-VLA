"""Aggregate per-suite LIBERO RLDS stats into a single shared Q99 payload.

Why: v34 multi-domain training collapsed (0/50 on libero_spatial) because each
of the 4 LIBERO suites was normalized with its OWN per-suite Q99, so the
shared backbone (LoRA, MLP-Pro head) saw 4 different action distributions
mixed together and could not converge. v35 added a ``shared_stats_path``
mechanism that overrides RLDS per-suite normalization, but the v35 default
(libero_spatial stats) saturates the wider-range suites (libero_goal/10).

This tool computes a *combined* LIBERO Q99 payload that all 4 suites fit
inside, eliminating saturation:

  combined.q01 = min over suites of suite.q01    # widest lower bound
  combined.q99 = max over suites of suite.q99    # widest upper bound
  combined.min = min over suites of suite.min
  combined.max = max over suites of suite.max
  combined.mean = weighted-avg by num_transitions
  combined.std  = pooled std (Welford-style aggregate variance)
  combined.mask = all-must-agree (raises if suites disagree)
  combined.num_transitions / num_trajectories = sum

The output JSON keeps the same {dataset_name: {action, proprio, ...}} wrapper
shape that ``RLDSLiberoDataset._resolve_shared_stats`` accepts (single-key
unwrap path).

Usage:
  uv run tools/aggregate_libero_stats.py \\
    --rlds-root /misc/dl00/takaki/vla-gemma-4/data/modified_libero_rlds \\
    --out outputs/libero_combined_q99/dataset_statistics.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List

DEFAULT_SUITES = (
    "libero_spatial_no_noops",
    "libero_object_no_noops",
    "libero_goal_no_noops",
    "libero_10_no_noops",
)


def _find_stats_file(rlds_root: Path, suite: str) -> Path:
    """Pick the most recently modified ``dataset_statistics_*.json`` under
    ``<rlds_root>/<suite>/1.0.0/``. RLDS keeps multiple cached stats files
    keyed by hashes of the data + transform pipeline; the newest one matches
    the current pipeline state."""
    suite_dir = rlds_root / suite / "1.0.0"
    if not suite_dir.is_dir():
        raise FileNotFoundError(f"suite dir not found: {suite_dir}")
    cands = sorted(
        suite_dir.glob("dataset_statistics_*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not cands:
        raise FileNotFoundError(
            f"no dataset_statistics_*.json under {suite_dir} — run RLDS once "
            f"to populate cache, or check the suite name"
        )
    return cands[0]


def _aggregate_field(field: str, suite_stats: List[Dict]) -> Dict:
    """Aggregate one of {'action', 'proprio'} across suites.

    Returns a dict with the same shape as the per-suite field
    ({mean, std, max, min, q01, q99, mask}) but combined.
    """
    # Gather num_transitions per suite as the weight basis.
    weights = [int(s.get("num_transitions", 0)) for s in suite_stats]
    total = sum(weights)
    if total <= 0:
        raise ValueError(f"num_transitions sum to {total}; check inputs")

    fields = [s[field] for s in suite_stats]
    dim = len(fields[0]["mean"])
    # Sanity: every suite has same dim.
    for i, f in enumerate(fields[1:], start=1):
        if len(f["mean"]) != dim:
            raise ValueError(
                f"{field} dim mismatch: suite[0]={dim}, suite[{i}]={len(f['mean'])}"
            )

    # Combined mean: weighted average by num_transitions.
    combined_mean = [
        sum(w * f["mean"][k] for w, f in zip(weights, fields)) / total
        for k in range(dim)
    ]

    # Combined std via pooled variance:
    #   var_global[k] = sum_i n_i * (s_i^2 + (m_i - M)^2) / sum_i n_i
    combined_std = []
    for k in range(dim):
        agg_var = 0.0
        for w, f in zip(weights, fields):
            m_i, s_i = f["mean"][k], f["std"][k]
            agg_var += w * (s_i * s_i + (m_i - combined_mean[k]) ** 2)
        agg_var /= total
        combined_std.append(math.sqrt(max(agg_var, 0.0)))

    # Range fields: take outer-bound across suites so all suites fit inside.
    combined_q01 = [min(f["q01"][k] for f in fields) for k in range(dim)]
    combined_q99 = [max(f["q99"][k] for f in fields) for k in range(dim)]
    combined_min = [min(f["min"][k] for f in fields) for k in range(dim)]
    combined_max = [max(f["max"][k] for f in fields) for k in range(dim)]

    # Mask must agree across suites IF the cache has masks at all.
    # Background: RLDS-cached stats files don't include ``mask`` — that field
    # is injected at runtime in ``make_dataset_from_rlds`` based on
    # ``action_normalization_mask`` (set in oxe/materialize.py per
    # ActionEncoding). All LIBERO suites use ActionEncoding.EEF_POS, so mask
    # is identical: [True]*6 + [False] for action (don't normalize gripper),
    # and proprio mask is created downstream. We default to LIBERO's known
    # mask shape when missing, and only enforce agreement if any suite
    # actually carries one.
    masks_from_cache = [f.get("mask") for f in fields]
    if all(m is None for m in masks_from_cache):
        if dim == 7:
            combined_mask = [True] * 6 + [False]   # LIBERO action: 6 deltas + gripper
        else:
            combined_mask = [True] * dim            # proprio: normalize all dims
    else:
        present = [tuple(bool(x) for x in m) for m in masks_from_cache if m is not None]
        if len(set(present)) != 1:
            raise ValueError(
                f"{field} mask disagrees across suites:\n"
                + "\n".join(f"  suite[{i}]: {m}" for i, m in enumerate(present))
            )
        combined_mask = list(present[0])

    return {
        "mean": combined_mean,
        "std": combined_std,
        "max": combined_max,
        "min": combined_min,
        "q01": combined_q01,
        "q99": combined_q99,
        "mask": combined_mask,
    }


def aggregate(
    rlds_root: Path,
    suites: List[str],
    out_path: Path,
    out_key: str,
) -> Dict:
    """Read each suite's cached stats, aggregate, write JSON, return payload."""
    suite_stats: List[Dict] = []
    suite_files: List[Path] = []
    for suite in suites:
        fp = _find_stats_file(rlds_root, suite)
        suite_files.append(fp)
        with open(fp, "r") as f:
            suite_stats.append(json.load(f))
        print(f"[aggregate] {suite}: {fp.name}")
        print(f"  num_transitions={suite_stats[-1].get('num_transitions')}")
        print(f"  num_trajectories={suite_stats[-1].get('num_trajectories')}")

    combined = {
        "action": _aggregate_field("action", suite_stats),
        "proprio": _aggregate_field("proprio", suite_stats),
        "num_transitions": sum(int(s.get("num_transitions", 0)) for s in suite_stats),
        "num_trajectories": sum(int(s.get("num_trajectories", 0)) for s in suite_stats),
    }
    payload = {out_key: combined}

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\n[aggregate] wrote combined stats → {out_path}")
    print(f"  source files:")
    for fp in suite_files:
        print(f"    {fp}")
    print(f"  combined.action.q99: {combined['action']['q99']}")
    print(f"  combined.action.q01: {combined['action']['q01']}")
    print(f"  combined.proprio.q99: {combined['proprio']['q99']}")
    print(f"  combined.proprio.q01: {combined['proprio']['q01']}")
    print(f"  num_transitions: {combined['num_transitions']}")
    print(f"  num_trajectories: {combined['num_trajectories']}")
    return payload


def main(argv: List[str]) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument(
        "--rlds-root",
        type=Path,
        default=Path("/misc/dl00/takaki/vla-gemma-4/data/modified_libero_rlds"),
        help="Root directory containing <suite>/1.0.0/dataset_statistics_*.json files.",
    )
    p.add_argument(
        "--suites",
        nargs="+",
        default=list(DEFAULT_SUITES),
        help="Suite names to aggregate (default: all 4 LIBERO RLDS suites).",
    )
    p.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output JSON path. Parent directory is created if missing.",
    )
    p.add_argument(
        "--out-key",
        type=str,
        default="libero_combined_no_noops",
        help="Top-level dict key under which the combined stats are stored "
        "(matches the wrapper-shape that RLDSLiberoDataset._resolve_shared_stats "
        "accepts via single-key unwrap).",
    )
    args = p.parse_args(argv)
    aggregate(args.rlds_root, args.suites, args.out, args.out_key)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
