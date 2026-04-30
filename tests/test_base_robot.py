"""Interface contract tests for BaseRobot.

We don't sim anything — just verify the abstract base enforces the four-
method protocol and that a tiny subclass can be instantiated and called.
"""
import pytest
from typing import Any, Dict

import numpy as np

from vla_project.robots.base_robot import BaseRobot


class _ToyRobot(BaseRobot):
    def __init__(self) -> None:
        self.connected = False
        self.last_action: np.ndarray | None = None

    def connect(self) -> None:
        self.connected = True

    def reset(self) -> Dict[str, Any]:
        return {
            "scene_image": np.zeros((4, 4, 3), dtype=np.uint8),
            "wrist_image": np.zeros((4, 4, 3), dtype=np.uint8),
            "proprio":     np.zeros(8, dtype=np.float32),
            "language":    "noop",
        }

    def get_observation(self) -> Dict[str, Any]:
        return self.reset()

    def send_action(self, action: np.ndarray) -> Dict[str, Any]:
        self.last_action = np.asarray(action, dtype=np.float32)
        return self.get_observation()

    def close(self) -> None:
        self.connected = False


def test_base_robot_is_abstract() -> None:
    with pytest.raises(TypeError):
        BaseRobot()  # type: ignore[abstract]


def test_subclass_round_trip() -> None:
    r = _ToyRobot()
    assert not r.connected
    r.connect()
    assert r.connected
    obs = r.reset()
    assert obs["scene_image"].shape == (4, 4, 3)
    assert obs["proprio"].shape == (8,)
    assert obs["language"] == "noop"

    r.send_action(np.array([0.1] * 7, dtype=np.float32))
    assert r.last_action is not None
    assert r.last_action.shape == (7,)

    r.close()
    assert not r.connected
