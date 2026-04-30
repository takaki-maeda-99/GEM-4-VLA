"""Integration smoke for LIBEROSimRobot.

This test boots a real LIBERO env (MuJoCo off-screen render). It is gated
by ``importlib.util.find_spec("libero")`` AFTER injecting LIBERO_PATH so
the test SKIPs cleanly on machines without the LIBERO repo on disk.

Wallclock: ~10-30 s on dl40 (env init dominates).
"""
import importlib.util
import os
import sys

import numpy as np
import pytest

from vla_project.robots.sim_robot import LIBEROSimRobot

# Resolve LIBERO_PATH and probe for libero importability before parametrizing.
_LIBERO_PATH = os.environ.get(
    "LIBERO_PATH", "/misc/dl00/takaki/vla-gemma-4/LIBERO"
)
if _LIBERO_PATH not in sys.path:
    sys.path.insert(0, _LIBERO_PATH)
_LIBERO_AVAILABLE = importlib.util.find_spec("libero") is not None


pytestmark = pytest.mark.skipif(
    not _LIBERO_AVAILABLE,
    reason=f"libero not importable from {_LIBERO_PATH}",
)


def test_connect_reset_step_close() -> None:
    """Reset + 10 dummy steps must yield finite obs every step."""
    robot = LIBEROSimRobot(
        bddl_path_root="/misc/dl00/takaki/vla-gemma-4/LIBERO/libero/libero/bddl_files",
        task_suite="libero_spatial",
        task_idx=0,
        image_size=224,
        seed=0,
        libero_path=_LIBERO_PATH,
    )
    robot.connect()
    try:
        obs = robot.reset()
        assert obs["scene_image"].shape == (224, 224, 3)
        assert obs["scene_image"].dtype == np.uint8
        assert obs["wrist_image"].shape == (224, 224, 3)
        assert obs["proprio"].shape == (8,)
        assert obs["proprio"].dtype == np.float32
        assert isinstance(obs["language"], str)
        assert obs["language"]  # non-empty

        for _ in range(10):
            obs = robot.send_action(np.zeros(7, dtype=np.float32))
            assert np.isfinite(obs["proprio"]).all()
            assert obs["scene_image"].shape == (224, 224, 3)
    finally:
        robot.close()


def test_get_observation_does_not_step() -> None:
    """Two consecutive get_observation() calls return data from the same
    sim time step (proprio identical)."""
    robot = LIBEROSimRobot(
        bddl_path_root="/misc/dl00/takaki/vla-gemma-4/LIBERO/libero/libero/bddl_files",
        task_suite="libero_spatial",
        task_idx=0,
        image_size=224,
        seed=0,
        libero_path=_LIBERO_PATH,
    )
    robot.connect()
    try:
        robot.reset()
        a = robot.get_observation()
        b = robot.get_observation()
        np.testing.assert_array_equal(a["proprio"], b["proprio"])
    finally:
        robot.close()
