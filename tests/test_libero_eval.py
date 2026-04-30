"""Stub-based test for evaluate_libero (no MuJoCo)."""
from typing import Any, Dict

import numpy as np

from vla_project.evaluation.libero_eval import evaluate_libero
from vla_project.policies.base_policy import BasePolicy
from vla_project.robots.base_robot import BaseRobot


class _FakePolicy(BasePolicy):
    def select_action(self, obs: Dict[str, Any]) -> np.ndarray:
        return np.zeros(7, dtype=np.float32)

    def reset(self) -> None:
        pass


class _FakeRobot(BaseRobot):
    """Reports success after step 5 deterministically."""

    def __init__(self, *_a, **_kw) -> None:
        self.t = 0
        self.connected = False

    def _obs(self):
        return {
            "scene_image": np.zeros((4, 4, 3), dtype=np.uint8),
            "wrist_image": np.zeros((4, 4, 3), dtype=np.uint8),
            "proprio":     np.zeros(8, dtype=np.float32),
            "language":    "fake",
        }

    def connect(self) -> None:
        self.connected = True

    def reset(self):
        self.t = 0
        return self._obs()

    def get_observation(self):
        return self._obs()

    def send_action(self, action):
        self.t += 1
        return self._obs()

    def check_success(self) -> bool:
        return self.t >= 5

    def close(self) -> None:
        self.connected = False


def test_evaluate_libero_runs_with_stub_robot_factory() -> None:
    p = _FakePolicy()

    def robot_factory(task_idx: int) -> BaseRobot:
        return _FakeRobot()

    out = evaluate_libero(
        policy=p,
        robot_factory=robot_factory,
        task_idxs=[0, 1, 2],
        num_episodes_per_task=2,
        max_steps=20,
        num_steps_wait=2,
        task_label_fn=lambda i: f"task_{i}",
    )
    assert out["overall"]["num_episodes"] == 6
    assert out["overall"]["num_success"] == 6  # every fake episode succeeds
    assert set(out["per_task"].keys()) == {"task_0", "task_1", "task_2"}
    for k in out["per_task"]:
        assert out["per_task"][k]["num_episodes"] == 2
        assert out["per_task"][k]["success_rate"] == 1.0
