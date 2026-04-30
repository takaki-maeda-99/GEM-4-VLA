"""Aggregation tests for evaluation.metrics."""
from vla_project.evaluation.metrics import aggregate_episodes
from vla_project.evaluation.rollout import EpisodeResult


def _ep(task: str, success: bool, steps: int = 100) -> dict:
    return {
        "task": task,
        "result": EpisodeResult(
            success=success,
            num_env_steps=steps,
            num_policy_calls=steps // 2,
            elapsed_s=1.0,
        ),
    }


def test_aggregates_per_task_and_overall() -> None:
    eps = [
        _ep("t0", True, 80),
        _ep("t0", False, 200),
        _ep("t1", True, 50),
        _ep("t1", True, 60),
    ]
    out = aggregate_episodes(eps)
    assert out["overall"]["num_episodes"] == 4
    assert out["overall"]["num_success"] == 3
    assert out["overall"]["success_rate"] == 0.75
    assert out["per_task"]["t0"]["num_episodes"] == 2
    assert out["per_task"]["t0"]["success_rate"] == 0.5
    assert out["per_task"]["t1"]["success_rate"] == 1.0
    assert out["per_task"]["t0"]["mean_env_steps"] == 140.0
    assert out["per_task"]["t1"]["mean_env_steps"] == 55.0


def test_empty_list_returns_zeroed_overall() -> None:
    out = aggregate_episodes([])
    assert out["overall"] == {
        "num_episodes": 0,
        "num_success": 0,
        "success_rate": 0.0,
        "mean_env_steps": 0.0,
    }
    assert out["per_task"] == {}
