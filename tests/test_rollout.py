"""Stub-based tests for rollout.run_episode.

We use a fake policy + fake robot — no MuJoCo, no model.
"""
from typing import Any, Dict

import numpy as np
import pytest

from vla_project.evaluation.rollout import EpisodeResult, run_episode
from vla_project.policies.base_policy import BasePolicy
from vla_project.robots.base_robot import BaseRobot


class _FakePolicy(BasePolicy):
    def __init__(self) -> None:
        self.calls = 0
        self.reset_calls = 0

    def select_action(self, obs: Dict[str, Any]) -> np.ndarray:
        self.calls += 1
        return np.zeros(7, dtype=np.float32)

    def reset(self) -> None:
        self.reset_calls += 1


class _FakeRobot(BaseRobot):
    """Reaches success after `succeed_at` env steps."""

    def __init__(self, succeed_at: int = -1) -> None:
        self.succeed_at = succeed_at
        self.t = 0
        self.connected = False
        self.last_action: np.ndarray | None = None

    def _obs(self) -> Dict[str, Any]:
        return {
            "scene_image": np.zeros((4, 4, 3), dtype=np.uint8),
            "wrist_image": np.zeros((4, 4, 3), dtype=np.uint8),
            "proprio":     np.zeros(8, dtype=np.float32),
            "language":    "fake",
        }

    def connect(self) -> None:
        self.connected = True

    def reset(self) -> Dict[str, Any]:
        self.t = 0
        return self._obs()

    def get_observation(self) -> Dict[str, Any]:
        return self._obs()

    def send_action(self, action: np.ndarray) -> Dict[str, Any]:
        self.t += 1
        self.last_action = np.asarray(action, dtype=np.float32)
        return self._obs()

    def check_success(self) -> bool:
        return 0 <= self.succeed_at <= self.t

    def close(self) -> None:
        self.connected = False


def test_loop_runs_full_length_when_no_success() -> None:
    p = _FakePolicy()
    r = _FakeRobot(succeed_at=-1)
    r.connect()
    out = run_episode(p, r, max_steps=20, num_steps_wait=5)
    r.close()
    assert isinstance(out, EpisodeResult)
    assert out.success is False
    # 5 warmup zero-action steps + 20 policy-action steps = 25 sim steps.
    assert out.num_env_steps == 25
    # Policy is called once per non-warmup step.
    assert p.calls == 20
    # Policy.reset() called once at the start of the episode.
    assert p.reset_calls == 1


def test_loop_short_circuits_on_success() -> None:
    p = _FakePolicy()
    r = _FakeRobot(succeed_at=8)  # success after step 8 (counted from sim 0)
    r.connect()
    out = run_episode(p, r, max_steps=100, num_steps_wait=5)
    r.close()
    assert out.success is True
    # 5 warmup + N policy steps where N is the smallest integer such that
    # warmup + N >= 8. So N = 3 → total 8 sim steps.
    assert out.num_env_steps == 8


def test_warmup_steps_zero_action() -> None:
    p = _FakePolicy()
    r = _FakeRobot(succeed_at=-1)
    r.connect()
    run_episode(p, r, max_steps=2, num_steps_wait=3)
    r.close()
    assert r.last_action is not None
    assert np.all(r.last_action == 0)


def test_zero_max_steps_is_warmup_only() -> None:
    p = _FakePolicy()
    r = _FakeRobot(succeed_at=-1)
    r.connect()
    out = run_episode(p, r, max_steps=0, num_steps_wait=3)
    r.close()
    assert out.num_env_steps == 3
    assert p.calls == 0
