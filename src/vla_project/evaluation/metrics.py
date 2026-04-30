"""Aggregate per-episode results into per-task / overall summary dicts."""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, Iterable, List

from vla_project.evaluation.rollout import EpisodeResult


def _summary(results: List[EpisodeResult]) -> Dict[str, Any]:
    n = len(results)
    if n == 0:
        return {"num_episodes": 0, "num_success": 0, "success_rate": 0.0, "mean_env_steps": 0.0}
    n_success = sum(1 for r in results if r.success)
    mean_steps = sum(r.num_env_steps for r in results) / n
    return {
        "num_episodes": n,
        "num_success": n_success,
        "success_rate": n_success / n,
        "mean_env_steps": float(mean_steps),
    }


def aggregate_episodes(records: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    """Group a flat list of ``{"task": str, "result": EpisodeResult}`` records
    into per-task summaries plus an overall summary."""
    records = list(records)
    overall = _summary([r["result"] for r in records])
    per_task: Dict[str, List[EpisodeResult]] = defaultdict(list)
    for r in records:
        per_task[r["task"]].append(r["result"])
    return {
        "overall": overall,
        "per_task": {task: _summary(rs) for task, rs in per_task.items()},
    }
